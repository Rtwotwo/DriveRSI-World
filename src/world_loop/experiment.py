"""One bounded RSI outer iteration: evidence, frozen rollout, PPO, paired evaluation.

This runner coordinates the World and Policy processes. It never imports Policy
source, exports a synthetic dataset, or promotes a candidate from one scene.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from driversi_world.simulation.closed_loop.environment import (
    ClosedLoopWorld,ENVIRONMENT_VERSION,ACTION_VERSION,CAMERAS,
)
from driversi_world.simulation.closed_loop.rewards import REWARD_VERSION
from driversi_world.runtime.paths import experiment_path
from .observations import ObservationHistory
from .policy_client import PolicyClient


def write_record(path, value):
    """One atomically replaced record; no proliferation of per-stage report files."""
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temporary.replace(path)


def source_snapshot():
    """Freeze the runner, observation adapter, simulator and focused test bytes."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root/'src/simulation/closed_loop').glob('*.py'))
    paths += list((root/'src/world_loop').glob('*.py'))
    paths += [root/p for p in (
        'src/simulation/render/mtgs/mtgs.py','src/simulation/render/mtgs/state.py',
        'src/simulation/envs/base_env.py','src/simulation/manager/agent_manager.py',
        'src/simulation/manager/render_manager.py','src/simulation/configs/default_runner.yaml',
        'tests/test_world_closed_loop.py','tests/test_world_ppo_budget.py')]
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def interventions(world):
    """Actual same-initial-state braking/advance interventions, including follower response."""
    branches,images,initial = {},{},{}
    for name,speed in [('brake',0.),('advance',3.)]:
        history = ObservationHistory(world,'intervention-'+name)
        history.warmup()
        initial[name] = dict(state=world.ego.controller.state.tolist(),
            hashes={c:hashlib.sha256(world.last['cameras'][c]['image'].tobytes()).hexdigest() for c in CAMERAS})
        rows = []
        for _ in range(4):
            rendered,reward,terminated,truncated,info = world.step(world.straight_action(speed))
            rows.append(dict(reward=reward,terminated=terminated,truncated=truncated,**info))
            if terminated or truncated:
                break
        branches[name] = rows
        images[name] = {c:world.last['cameras'][c]['image'].copy() for c in CAMERAS}
    assert initial['brake']==initial['advance'],'Counterfactuals must start from identical state and images'
    assert len(branches['brake'])==len(branches['advance']),'Interventions need a common comparison time'
    left,right = branches['brake'][-1],branches['advance'][-1]
    displacement = float(np.linalg.norm(np.array(left['state'][:2])-right['state'][:2]))
    difference = {c:float(np.abs(images['brake'][c].astype(float)-images['advance'][c]).mean()) for c in CAMERAS}
    reward_difference = float(abs(sum(r['reward'] for r in branches['brake'])-sum(r['reward'] for r in branches['advance'])))
    followers = {}
    for key,a in left['traffic'].items():
        if key not in right['traffic']:
            continue
        b = right['traffic'][key]
        saw_ego = any(row['traffic'].get(key,{}).get('leader')=='ego' for rows in branches.values() for row in rows)
        delta = float(np.linalg.norm(np.array(a['position'])-b['position']))
        if saw_ego:
            followers[key] = dict(position_difference_m=delta,speed_difference_mps=abs(a['next_speed_mps']-b['next_speed_mps']))
    checks = dict(action_changes_state=displacement>1e-3,
                  action_changes_observation=all(v>.01 for v in difference.values()),
                  action_changes_reward=reward_difference>1e-5,
                  reactive_traffic_verified=any(v['position_difference_m']>1e-5 for v in followers.values()))
    import cv2
    panels = []
    for c in CAMERAS:
        tiles = []
        for name in ('brake','advance'):
            tile = cv2.resize(images[name][c],(768,432),interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile,(0,0),(768,32),(0,0,0),-1)
            cv2.putText(tile,f'{name} | {c} | executed step {world.step_index}',(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            tiles.append(tile)
        panels.append(np.concatenate(tiles,axis=1))
    if not cv2.imwrite(str(Path(world.config['output'])/'intervention.png'),np.concatenate(panels,axis=0)):
        raise RuntimeError('Could not write intervention evidence')
    result = dict(checks=checks,identical_initial_state=True,ego_displacement_m=displacement,
        image_mae_0_255=difference,reward_sum_difference=reward_difference,followers=followers,branches=branches)
    if not all(checks.values()):
        raise RuntimeError('One or more real action/reaction checks failed: '+json.dumps(result))
    return result


def policy_episode(world, policy, name, training, max_steps=None, episode_seed=None, phase=None):
    history = ObservationHistory(world,name)
    observation = history.warmup()
    if episode_seed is not None:
        response=policy.call('begin_episode',episode_id=name,seed=episode_seed,phase=phase)
        if any(response.get(k)!=v for k,v in dict(episode_id=name,seed=episode_seed,phase=phase).items()):
            raise ValueError('Policy did not acknowledge the exact episode seed and phase')
    # Compare physical initial conditions and pixel contents, not IPC file paths:
    # each episode deliberately has a fresh sensor directory.
    participants={key:dict(position=np.asarray(actor['position']).tolist(),
                          **{k:float(actor[k]) for k in ('heading','speed','length','width')})
                  for key,actor in world.participants().items()}
    initial=dict(state=world.ego.controller.state.tolist(),timestamp=observation['frames'][-1]['timestamp'],
        traffic_sha256=hashlib.sha256(json.dumps(participants,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        images_sha256={c:hashlib.sha256(world.last['cameras'][c]['image'].tobytes()).hexdigest() for c in CAMERAS})
    rows = []
    while True:
        start = time.perf_counter()
        action = policy.call('act',observation=observation,deterministic=not training,record=training)
        action_seconds = time.perf_counter()-start
        if action['dt_seconds']!=.5:
            raise ValueError('Policy action time step differs')
        native = world.trajectory(action['trajectory_ego'])
        conversion_error = float(np.max(np.abs(native.waypoints-np.asarray(action['trajectory_world']['waypoints']))))
        if conversion_error>1e-4:
            raise ValueError('Policy and World disagree on ego/rear/center coordinates')
        rendered,reward,terminated,truncated,execution = world.step(action['trajectory_ego'])
        if max_steps is not None and len(rows)+1>=max_steps and not terminated:
            truncated = True
        history.append(rendered)
        observation = history.payload()
        if training:
            accepted = policy.call('transition',action_id=action['action_id'],reward=reward,
                                  terminated=terminated,truncated=truncated,next_observation=observation,execution=execution)
            if accepted.get('accepted') is not True:
                raise RuntimeError('Executed transition was not accepted by Policy')
        rows.append(dict(action_id=action['action_id'],policy_version=action['policy_version'],
                         reward=reward,terminated=terminated,truncated=truncated,
                         conversion_error_m=conversion_error,action_seconds=action_seconds,**execution))
        print('STEP',name,world.step_index,'reward',round(reward,6),'terminal',terminated,'truncated',truncated,flush=True)
        if terminated or truncated:
            break
    return dict(initial=initial,episode_seed=episode_seed,phase=phase,rows=rows,reward_sum=sum(r['reward'] for r in rows),
        steps=len(rows),collisions=sum(bool(r['metrics']['collisions']) for r in rows),
        maximum_offroad_fraction=max(r['metrics']['offroad_fraction'] for r in rows),
        progress_m=sum(r['metrics']['delta_progress_m'] for r in rows),
        max_lateral_m=max(r['metrics']['lateral_m'] for r in rows))


def collect_training(world,policy,budget,seeds=None,phase='training'):
    """Collect exactly the budget, even when collisions terminate short episodes.

    All samples come from the current frozen behavior policy. Terminal samples
    are retained for PPO; resetting the environment does not erase a failure.
    """
    if type(budget) is not int or not 4 <= budget <= 64:
        raise ValueError('PPO training budget must be 4..64 executed transitions')
    episodes,remaining = [],budget
    while remaining:
        kwargs={}
        if seeds is not None:
            if len(episodes)>=len(seeds):
                raise ValueError('Declared episode seeds exhausted before the transition budget')
            kwargs=dict(episode_seed=seeds[len(episodes)],phase=phase)
        result=policy_episode(world,policy,f'{phase}-{len(episodes)}',True,max_steps=min(8,remaining),**kwargs)
        if not 1 <= result['steps'] <= remaining:
            raise RuntimeError('Episode did not respect the remaining rollout budget')
        episodes.append(result)
        remaining -= result['steps']
    return episodes


def evaluate_policy(world,policy,name,steps,seeds):
    """Match parent/child seeds; stochastic policy seeds do not create new scenes."""
    if seeds is None:
        return policy_episode(world,policy,name,False,max_steps=steps)
    episodes=[policy_episode(world,policy,f'{name}-{i}',False,max_steps=steps,
              episode_seed=seed,phase='evaluation') for i,seed in enumerate(seeds)]
    rows=[row for episode in episodes for row in episode['rows']]
    return dict(episodes=episodes,rows=rows,steps=len(rows),
        initial=[episode['initial'] for episode in episodes],
        reward_sum=sum(episode['reward_sum'] for episode in episodes),
        progress_m=sum(episode['progress_m'] for episode in episodes),
        mean_reward=float(np.mean([episode['reward_sum'] for episode in episodes])),
        mean_progress_m=float(np.mean([episode['progress_m'] for episode in episodes])),
        collisions=sum(episode['collisions'] for episode in episodes),
        maximum_offroad_fraction=max(episode['maximum_offroad_fraction'] for episode in episodes),
        max_lateral_m=max(episode['max_lateral_m'] for episode in episodes),
        scope='multiple policy seeds on one identical physical scene')


def training_plan(config):
    """Require World budgets and the frozen Policy recipe to agree before loading."""
    path=config.get('policy_training_config')
    if path is None:
        if config.get('requires_policy_training_config'):
            raise ValueError('Agent task must supply the frozen Policy training config')
        if config.get('training_steps',8)>32:
            raise ValueError('More than 32 transitions requires an explicit training config')
        return None
    plan=json.loads(Path(path).read_text())
    expected=dict(schema='driversi.ppo-training.v1',pilot_transitions=32,actor_epochs=2)
    keys=set(expected)|{'batch_transitions','critic_warmup_epochs'}
    if (not isinstance(plan,dict) or set(plan)!=keys or
            any(plan.get(k)!=v for k,v in expected.items()) or
            any(type(plan[k]) is not int for k in keys-{'schema'})):
        raise ValueError('Unsupported Policy pilot or actor protocol')
    if (plan.get('batch_transitions') not in (16,64) or type(plan.get('critic_warmup_epochs')) is not int
            or plan['critic_warmup_epochs'] not in (0,32) or config.get('training_steps')!=plan['batch_transitions']
            or config.get('pilot_steps')!=32):
        raise ValueError('World and Policy sample budgets disagree')
    declared=[]
    for key,count,maximum in [('evaluation_seeds',3,3),('pilot_seeds',32,32),
                              ('training_seeds',plan['batch_transitions'],64)]:
        seeds=config.get(key)
        if (not isinstance(seeds,list) or not count<=len(seeds)<=maximum or
                any(type(s) is not int or not 0<=s<2**31 for s in seeds) or len(set(seeds))!=len(seeds)):
            raise ValueError('Distinct, sufficient, bounded '+key+' required')
        declared.extend(seeds)
    if len(set(declared))!=len(declared):
        raise ValueError('Pilot, training and evaluation seed lists must be disjoint')
    return dict(plan,path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def check_policy_plan(status, plan):
    """Require the resident service to acknowledge the exact frozen experiment."""
    if plan is None:
        return
    required={'begin_episode_v1','separate_value_pilot_v1'}
    recipe={k:v for k,v in plan.items() if k not in ('path','sha256')}
    if (not required.issubset(status.get('capabilities',[])) or
            status.get('training_config')!=recipe or status.get('policy_version')!=0 or
            status.get('updates')!=0 or status.get('transitions')!=0 or
            status.get('pending_action') is not None or status.get('value_warmup_complete') is not False):
        raise ValueError('Policy service did not acknowledge a fresh matching experiment')


def run(config_path, probe_only=False):
    config = json.loads(Path(config_path).read_text())
    if config.get('challenge') is not None:
        raise ValueError('Challenge candidates require a dedicated qualified evaluator; this PPO runner admits original scenes only')
    evaluation_steps=config.get('evaluation_steps',4)
    training_steps=config.get('training_steps',8)
    if type(evaluation_steps) is not int or not 1 <= evaluation_steps <= 8:
        raise ValueError('Evaluation horizon must fit the eight post-warmup steps')
    if type(training_steps) is not int or not 4 <= training_steps <= 64:
        raise ValueError('PPO training budget must be 4..64 executed transitions')
    plan=training_plan(config)
    out = experiment_path(config['output'])
    out.mkdir(parents=True,exist_ok=True)
    record_path = out/'result.json'
    core = Path(__file__).parents[1]/'simulation/closed_loop'
    code_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')}
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    record = dict(schema='driversi.world-outer-loop.v1',status='running',config=config,
        flash_enabled=False,dataset_exported=False,model_promoted=False,
        frozen_versions=dict(environment=ENVIRONMENT_VERSION,reward=REWARD_VERSION,action=ACTION_VERSION),
        observation_version='rear-ego-inertial-v2',
        limitations=['Single scene integration, not a driving capability evaluation',
                     'Vehicle IDM is longitudinal only; no lane-change or traffic-light planner',
                     'Progress reward uses a fixed reference route extracted from the full log; it is not exposed to the policy',
                     'Collision/offroad checks sample 0.5-second endpoints and can miss inter-step contact',
                     'Pedestrian/cyclist collision boxes follow logs; their appearance is not independently reconstructed'])
    record['world_code_sha256'] = code_hashes
    record['training_plan']=plan
    record['source_sha256_at_launch'] = source_snapshot()
    record['previous_attempts'] = (previous or {}).get('previous_attempts', [])
    if previous and previous.get('status') == 'failed':
        record['previous_attempts'].append(dict(
            policy_output=previous['config']['policy_output'],error=previous.get('error'),
            policy_initial=previous.get('policy_initial'),
            completed_training_steps=sum(x['steps'] for x in previous.get('training',[]))))
    world,policy = None,None
    try:
        world = ClosedLoopWorld(config)
        same_scene = previous and all(previous.get('config',{}).get(k)==config[k]
                                      for k in ('scene','scenarios','maps','asset_root'))
        flags=('action_changes_state','action_changes_observation','action_changes_reward','reactive_traffic_verified')
        if same_scene and previous.get('world_code_sha256')==code_hashes and all(previous.get('interventions',{}).get('checks',{}).get(k) is True for k in flags):
            record['interventions'] = previous['interventions']
            record['used_verified_intervention_record'] = True
        else:
            record['interventions'] = interventions(world)
        write_record(record_path,record)
        print('REAL_INTERVENTIONS_PASSED',flush=True)
        if probe_only:
            record['status'] = 'interventions_passed'
            return
        policy = PolicyClient(config)
        record['policy_initial'] = policy.call('status')
        check_policy_plan(record['policy_initial'],plan)
        record['evaluation_horizon_steps'] = evaluation_steps
        seeds=config.get('evaluation_seeds') if plan else None
        record['parent_evaluation'] = evaluate_policy(world,policy,'parent-evaluation',evaluation_steps,seeds)
        if plan:
            record['pilot']=collect_training(world,policy,32,config['pilot_seeds'],phase='pilot')
            before=policy.call('status')
            if before['transitions']!=32:
                raise RuntimeError('Policy did not retain the exact 32-step pilot')
            record['critic_warmup']=policy.call('warmup_value')
            after=policy.call('status')
            if (before['decoder_sha256']!=after['decoder_sha256'] or before['policy_version']!=after['policy_version']
                    or after['transitions']!=0 or after['pending_action'] is not None or after['updates']!=0
                    or after.get('value_warmup_complete') is not True):
                raise RuntimeError('Pilot warmup changed the actor or left training samples pending')
            record['pilot_discard_verified']=True
            write_record(record_path,record)
        record['training'] = collect_training(world,policy,training_steps,
                                             config['training_seeds'] if plan else None)
        write_record(record_path,record)
        record['update'] = policy.call('update')
        if not record['update']['decoder_changed'] or record['update']['committed_epochs']<1:
            raise RuntimeError('PPO did not commit a real actor parameter change')
        record['child_evaluation'] = evaluate_policy(world,policy,'child-evaluation',evaluation_steps,seeds)
        record['paired_evaluation_initial_conditions_equal']=(
            record['parent_evaluation']['initial']==record['child_evaluation']['initial'])
        if not record['paired_evaluation_initial_conditions_equal']:
            raise RuntimeError('Parent and child evaluation initial state, traffic, or cameras differ')
        record['policy_final'] = policy.call('status')
        policy.close();policy=None
        if plan and hashlib.sha256(Path(plan['path']).read_bytes()).hexdigest()!=plan['sha256']:
            raise RuntimeError('Policy training recipe changed during the run')
        record['source_unchanged_during_run'] = source_snapshot()==record['source_sha256_at_launch']
        if not record['source_unchanged_during_run']:
            raise RuntimeError('World source changed while sampling or updating')
        session = Path(config['policy_output'])/'result.json'
        record['policy_handoff'] = dict(schema='driversi.world-ppo-handoff.v1',policy_session=str(session),
            policy_session_sha256=hashlib.sha256(session.read_bytes()).hexdigest(),
            environment_version=ENVIRONMENT_VERSION,reward_version=REWARD_VERSION,
            action_transform_version=ACTION_VERSION,**record['interventions']['checks'])
        # A one-scene integration must not promote a model or change the reward to
        # force an improvement. The outer loop explicitly keeps the current release.
        record['outer_decision'] = dict(action='retain_release',candidate_trained=True,
            reason='Integration evidence only; independent held-out capability and regression evaluation absent',
            next_allowed_action='expand fixed-recipe closed-loop evaluation before curriculum or environment changes')
        record['status'] = 'complete'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        if world is not None:
            record['executed_environment_steps']=world.executed_environment_steps
            record['environment_resets']=world.episode
        write_record(record_path,record)
        if policy is not None:
            policy.close(graceful=False)
        if world is not None:
            world.close()


if __name__=='__main__':
    run(sys.argv[1],probe_only='--probe-only' in sys.argv)
