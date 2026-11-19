"""PPO must retain failures and respect a fixed transition budget across resets."""
import json
import pytest
from driversi_world.world_loop import experiment


def test_short_terminal_episodes_preserved_without_budget_overshoot(monkeypatch):
    remaining=[]
    def episode(world,policy,name,training,max_steps):
        remaining.append(max_steps)
        n=min(3,max_steps)
        return dict(steps=n,rows=[dict(terminated=True,truncated=False,reward=-10.)])
    monkeypatch.setattr(experiment,'policy_episode',episode)
    episodes=experiment.collect_training(None,None,8)
    assert remaining==[8,5,2]
    assert sum(x['steps'] for x in episodes)==8
    assert all(x['rows'][-1]['reward']==-10 for x in episodes)


@pytest.mark.parametrize('budget',[True,0,3,65,8.5])
def test_invalid_training_budget_rejected_before_rollout(budget):
    with pytest.raises(ValueError):experiment.collect_training(None,None,budget)


def test_empty_episode_fails_instead_of_looping_forever(monkeypatch):
    monkeypatch.setattr(experiment,'policy_episode',lambda *a,**k:dict(steps=0))
    with pytest.raises(RuntimeError):experiment.collect_training(None,None,8)


@pytest.mark.parametrize('key,value',[('evaluation_steps',9),('evaluation_steps',True),
                                    ('training_steps',33)])
def test_invalid_horizon_stops_before_world_or_policy_allocation(tmp_path,key,value):
    path=tmp_path/'config.json';path.write_text(json.dumps({key:value}))
    with pytest.raises(ValueError):experiment.run(path)


def plan_config(tmp_path,batch=64):
    path=tmp_path/'policy.json'
    path.write_text(json.dumps(dict(schema='driversi.ppo-training.v1',batch_transitions=batch,
        pilot_transitions=32,critic_warmup_epochs=32,actor_epochs=2)))
    return dict(policy_training_config=str(path),training_steps=batch,pilot_steps=32,
        evaluation_seeds=[36117,36118,36119],pilot_seeds=list(range(46117,46149)),
        training_seeds=list(range(56117,56181)))


def test_declared_seed_prefix_pairs_small_and_large_actor_batches(monkeypatch):
    calls=[]
    def episode(world,policy,name,training,max_steps,episode_seed,phase):
        calls.append((episode_seed,phase,max_steps))
        return dict(steps=max_steps)
    monkeypatch.setattr(experiment,'policy_episode',episode)
    seeds=list(range(56117,56181))
    small=experiment.collect_training(None,None,16,seeds)
    first=list(calls);calls.clear()
    large=experiment.collect_training(None,None,64,seeds)
    assert len(small)==2 and len(large)==8 and calls[:2]==first
    assert sum(x['steps'] for x in large)==64


def test_episode_seed_budget_is_bounded_under_early_termination(monkeypatch):
    monkeypatch.setattr(experiment,'policy_episode',lambda *a,**k:dict(steps=1))
    with pytest.raises(ValueError,match='seeds exhausted'):
        experiment.collect_training(None,None,16,[1,2])


def test_pilot_and_training_recipe_checked_before_gpu_allocation(tmp_path):
    config=plan_config(tmp_path)
    plan=experiment.training_plan(config)
    assert plan['batch_transitions']==64 and len(plan['sha256'])==64
    config['training_steps']=16
    with pytest.raises(ValueError,match='budgets disagree'):experiment.training_plan(config)


@pytest.mark.parametrize('edit',['overlap','duplicate','short','bool_seed','seed_overflow'])
def test_invalid_seed_schedules_fail_closed(tmp_path,edit):
    config=plan_config(tmp_path)
    if edit=='overlap':config['evaluation_seeds'][0]=config['pilot_seeds'][0]
    if edit=='duplicate':config['training_seeds'][1]=config['training_seeds'][0]
    if edit=='short':config['pilot_seeds']=[0]
    if edit=='bool_seed':config['evaluation_seeds'][0]=True
    if edit=='seed_overflow':config['evaluation_seeds'][0]=2**31
    with pytest.raises(ValueError):experiment.training_plan(config)


def test_multi_episode_evaluation_is_explicitly_one_scene(monkeypatch):
    def episode(*args,episode_seed,**kwargs):
        return dict(initial={'ego':0},episode_seed=episode_seed,rows=[{}],steps=1,
                    reward_sum=float(episode_seed),progress_m=1.,collisions=0,
                    maximum_offroad_fraction=0.,max_lateral_m=.1)
    monkeypatch.setattr(experiment,'policy_episode',episode)
    result=experiment.evaluate_policy(None,None,'eval',8,[10,20,30])
    assert result['steps']==3 and result['reward_sum']==60 and result['mean_reward']==20
    assert [e['episode_seed'] for e in result['episodes']]==[10,20,30]
    assert 'one identical physical scene' in result['scope']


@pytest.mark.parametrize('edit',['float_budget','extra_key','wrong_schema'])
def test_training_recipe_matches_policy_strict_types_and_keys(tmp_path,edit):
    config=plan_config(tmp_path)
    path=tmp_path/'policy.json';value=json.loads(path.read_text())
    if edit=='float_budget':value['batch_transitions']=64.
    if edit=='extra_key':value['future_option']=True
    if edit=='wrong_schema':value['schema']='unknown'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):experiment.training_plan(config)


@pytest.mark.parametrize('batch,warmup,fault',[
    (16,0,None),(16,32,None),(64,0,None),(64,32,None),
    (16,32,'actor_changed'),(16,32,'pilot_retained'),(16,32,'old_service')])
def test_cpu_runner_orders_pilot_fresh_actor_and_paired_evaluation(tmp_path,monkeypatch,batch,warmup,fault):
    """Exercise orchestration with an explicit fake service, without claiming GPU validation."""
    config=plan_config(tmp_path,batch)
    recipe_path=tmp_path/'policy.json'
    recipe=json.loads(recipe_path.read_text());recipe['critic_warmup_epochs']=warmup
    recipe_path.write_text(json.dumps(recipe))
    output=tmp_path/'world';policy_output=tmp_path/'policy'
    policy_output.mkdir()
    config.update(output=str(output),policy_output=str(policy_output),evaluation_steps=8,
                  scene='fixture',scenarios='fixture',maps='fixture',asset_root='fixture')
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    events=[]

    class World:
        def __init__(self,config):
            self.executed_environment_steps=0;self.episode=0;self.closed=False
        def close(self):self.closed=True;events.append('world-close')

    class Policy:
        def __init__(self,config):
            self.count=0;self.version=0;self.warmed=False;self.actor='parent'
        def call(self,op,**kwargs):
            if op=='status':
                return dict(transitions=self.count,policy_version=self.version,updates=self.version,
                    pending_action=None,decoder_sha256=self.actor,value_warmup_complete=self.warmed,
                    capabilities=[] if fault=='old_service' else ['begin_episode_v1','separate_value_pilot_v1'],
                    training_config=recipe)
            if op=='warmup_value':
                assert self.count==32 and self.version==0
                events.append('warmup');self.warmed=True
                if fault=='actor_changed':self.actor='wrong'
                if fault!='pilot_retained':self.count=0
                return dict(epochs=warmup)
            if op=='update':
                assert self.count==batch and self.warmed
                events.append('update');self.version=1;self.actor='child'
                return dict(decoder_changed=True,committed_epochs=2)
            raise AssertionError(op)
        def close(self,graceful=True):
            events.append('policy-close')
            (policy_output/'result.json').write_text('{}')

    def episode(world,policy,name,training,max_steps,episode_seed,phase):
        # Both training phases need new episode calls; a retained pilot cannot
        # silently count toward the actor batch. Parent/child share only eval seeds.
        events.append((phase,episode_seed,max_steps))
        world.episode+=1;world.executed_environment_steps+=3+max_steps
        if training:
            assert phase==('training' if policy.warmed else 'pilot')
            policy.count+=max_steps
        rows=[dict(reward=1.,metrics=dict(collisions=[],offroad_fraction=0.,delta_progress_m=1.,lateral_m=0.))
              for _ in range(max_steps)]
        return dict(initial={'state':[0.], 'pixels':'same'},episode_seed=episode_seed,rows=rows,
            steps=max_steps,reward_sum=float(max_steps),progress_m=float(max_steps),collisions=0,
            maximum_offroad_fraction=0.,max_lateral_m=0.)

    monkeypatch.setattr(experiment,'ClosedLoopWorld',World)
    monkeypatch.setattr(experiment,'PolicyClient',Policy)
    monkeypatch.setattr(experiment,'policy_episode',episode)
    monkeypatch.setattr(experiment,'experiment_path',lambda value:__import__('pathlib').Path(value))
    monkeypatch.setattr(experiment,'source_snapshot',lambda:{'fixture':'unchanged'})
    monkeypatch.setattr(experiment,'interventions',lambda world:dict(checks={}))
    if fault:
        with pytest.raises((RuntimeError,ValueError)):experiment.run(path)
        assert 'update' not in events
    else:
        experiment.run(path)
    result=json.loads((output/'result.json').read_text())
    assert events[-2:]==['policy-close','world-close']
    if fault:
        assert result['status']=='failed'
        return
    assert result['status']=='complete' and result['pilot_discard_verified']
    assert sum(e['steps'] for e in result['pilot'])==32
    assert sum(e['steps'] for e in result['training'])==batch
    assert result['parent_evaluation']['steps']==result['child_evaluation']['steps']==24
    assert result['paired_evaluation_initial_conditions_equal']
    assert result['executed_environment_steps']==(4+batch//8+6)*11
    assert events.index('warmup')<events.index('update')
    pilot=[e[1] for e in events if isinstance(e,tuple) and e[0]=='pilot']
    actor=[e[1] for e in events if isinstance(e,tuple) and e[0]=='training']
    evaluation=[e[1] for e in events if isinstance(e,tuple) and e[0]=='evaluation']
    assert set(pilot).isdisjoint(actor) and evaluation[:3]==evaluation[3:]
