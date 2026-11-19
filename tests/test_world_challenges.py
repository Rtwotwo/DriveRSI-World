"""Challenge admission and schedule tests; CPU results are never PPO evidence."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from driversi_world.simulation.closed_loop.challenges import (
    catalogue,validate,receipt,target_speed,apply_to_policies,digest,
)
from driversi_world.simulation.closed_loop.traffic import ReactiveIDMPolicy
from driversi_world.world_loop.challenger import candidate_requests,qualify_pair
from driversi_world.world_loop.challenge import smoke


def fixture():
    state=dict(position=np.column_stack([np.arange(24.),np.ones(24)*10,np.zeros(24)]),
        heading=np.zeros(24),velocity=np.column_stack([np.ones(24)*3,np.zeros(24)]),
        valid=np.ones(24),length=np.ones(24)*4,width=np.ones(24)*2)
    ego=copy.deepcopy(state);ego['position'][:,1]=0
    scene=dict(id='test-scene',object_track={'npc':dict(type='VEHICLE',state=state),
                                          'ego':dict(type='VEHICLE',state=ego)})
    asset={'rigid_object_npc':dict(config={'type':'MirroredRigidSubModel'},
                                 state_dict={'instance_trans':np.ones((24,3))})}
    cap=catalogue(scene,asset,'a'*64)
    request=candidate_requests(cap,11,1)[0]
    return scene,asset,cap,request


def test_catalogue_rejects_missing_static_and_disappearing_gs_actors():
    scene,asset,cap,_=fixture()
    assert set(cap['actors']) == {'npc'}
    assert catalogue(scene,{},'a'*64)['actors']=={}
    asset['rigid_object_npc']['state_dict']['instance_trans']=np.ones(3)
    assert catalogue(scene,asset,'a'*64)['actors']=={}
    scene,asset,_,_=fixture()
    scene['object_track']['npc']['state']['valid'][7]=0
    assert catalogue(scene,asset,'a'*64)['actors']=={}


def test_request_is_defensive_copy_and_receipt_is_not_training_approval():
    _,_,cap,request=fixture()
    new=validate(request,cap);new['edit']['desired_speed_scale']=1.5
    assert request['edit']['desired_speed_scale']==.5
    result=receipt(request,cap)
    assert result['status']=='structurally_valid' and not result['training_ready']
    assert result['actual_edit_result'] is None and result['reference_validation']=='pending'


@pytest.mark.parametrize('field,value',[
    ('actor_id','ego'),('actor_id','missing'),('desired_speed_scale',float('nan')),
    ('desired_speed_scale',0),('desired_speed_scale',2),('desired_speed_scale',True),
    ('start_step',2),('start_step',True),('duration_steps',0),('duration_steps',9),
])
def test_bad_edit_is_rejected(field,value):
    _,_,cap,request=fixture();request['edit'][field]=value
    with pytest.raises(ValueError):validate(request,cap)


@pytest.mark.parametrize('field,value',[
    ('split','test'),('scene_id','another-scene'),('scene_sha256','b'*64),
    ('capability_sha256','b'*64),('seed',True),('request_id','../foreign'),
])
def test_wrong_identity_or_split_is_rejected(field,value):
    _,_,cap,request=fixture();request[field]=value
    with pytest.raises(ValueError):validate(request,cap)


def test_unreachable_target_fields_and_initial_overlap_are_rejected():
    scene,asset,cap,request=fixture();request['edit']['end_position']=[1000,1000]
    with pytest.raises(ValueError):validate(request,cap)
    scene['object_track']['ego']['state']['position'][0]=scene['object_track']['npc']['state']['position'][0]
    overlapping=catalogue(scene,asset,'a'*64)
    request.pop('edit');request['edit']=dict(actor_id='npc',desired_speed_scale=.5,start_step=3,duration_steps=3)
    request['capability_sha256']=overlapping['capability_sha256']
    with pytest.raises(ValueError,match='overlap'):validate(request,overlapping)


def test_schedule_restores_nominal_without_changing_velocity_or_other_actor():
    _,_,_,request=fixture()
    assert [target_speed(4.,request,'npc',k) for k in range(8)]==[4,4,4,2,2,2,4,4]
    assert target_speed(4.,request,'other',4)==4
    actor=SimpleNamespace(policy=SimpleNamespace(nominal_desired_speed=4.,desired_speed=4.),velocity=3.)
    apply_to_policies({'npc':actor},request,3)
    assert actor.policy.desired_speed==2 and actor.velocity==3
    apply_to_policies({'npc':actor},request,6)
    assert actor.policy.desired_speed==4


def test_real_idm_policy_consumes_scheduled_target_and_keeps_si_bounds():
    # Exercise production act() without renderer: an analytical straight lane,
    # one live actor, and the same pre-step snapshot interface as ClosedLoopWorld.
    _,_,_,request=fixture()
    lane=SimpleNamespace(length=100.,local_coordinates=lambda p:(p[0],p[1]),
                         heading_theta_at=lambda s:0.,position=lambda s,l:np.array([s,l]))
    actor=SimpleNamespace(id='npc',navigation=SimpleNamespace(current_lane=lane),current_speed=3.)
    policy=object.__new__(ReactiveIDMPolicy)
    policy.agent=actor;policy.desired_speed=4.;policy.nominal_desired_speed=4.
    actor.policy=policy
    # BasePolicy.engine is obtained from the global engine; patch its class
    # property for this isolated CPU test rather than instantiate a CUDA engine.
    snapshot={'npc':dict(position=[0,0],heading=0.,length=4.,width=2.,speed=3.)}
    from unittest.mock import patch,PropertyMock
    with patch.object(ReactiveIDMPolicy,'engine',new_callable=PropertyMock,create=True) as engine:
        engine.return_value=SimpleNamespace(closed_loop_snapshot=snapshot)
        baseline=policy.act()
        apply_to_policies({'npc':actor},request,3)
        edited=policy.act()
        assert 0 <= np.linalg.norm(edited.velocities[1]) < np.linalg.norm(baseline.velocities[1])
        assert -4 <= policy.last_decision['acceleration'] <= 2
        assert edited.waypoints[1,0]>=0


def paired_evidence(request):
    reference=dict(request_sha256=digest(request),initial_state_sha256='i'*64,
        execution_contract_sha256='e'*64,policy_sha256='r'*64,seed=request['seed'],
        executed_steps=8,complete=True,collision_count=0,max_offroad_fraction=0.,progress_m=3.)
    current=dict(reference,policy_sha256='c'*64,progress_m=1.)
    validation=dict(trace_hashes_verified=True,training_split_verified=True,geometry_valid=True,rendering_valid=True)
    return current,reference,validation


def test_regret_requires_reference_solvability_and_matched_complete_evidence():
    _,_,_,request=fixture();cur,ref,gates=paired_evidence(request)
    result=qualify_pair(request,cur,ref,gates)
    assert result['eligible'] and result['score']==pytest.approx(2/3)
    ref['collision_count']=1
    assert not qualify_pair(request,cur,ref,gates)['training_ready']
    ref['seed']=20
    with pytest.raises(ValueError):qualify_pair(request,cur,ref,gates)


@pytest.mark.parametrize('gate',['trace_hashes_verified','training_split_verified','geometry_valid','rendering_valid'])
def test_cpu_or_unverified_evidence_cannot_enter_training(gate):
    _,_,_,request=fixture();cur,ref,gates=paired_evidence(request);gates[gate]=False
    result=qualify_pair(request,cur,ref,gates)
    assert result['score'] is None and not result['training_ready']


def test_cpu_smoke_responds_to_edit_but_does_not_claim_closed_loop():
    _,_,cap,request=fixture();result=smoke(cap,[request])
    assert result['trials'][0]['final_distance_difference_m']<0
    assert not result['rendered'] and not result['training_ready'] and not result['full_traffic_simulation']


def test_partial_success_is_not_a_complete_reference_or_current_rollout():
    _,_,_,request=fixture();cur,ref,gates=paired_evidence(request)
    cur['executed_steps']=3
    with pytest.raises(ValueError,match='incomplete'):qualify_pair(request,cur,ref,gates)


def test_excess_progress_cannot_hide_collision_from_challenger_score():
    _,_,_,request=fixture();cur,ref,gates=paired_evidence(request)
    cur.update(collision_count=1,progress_m=1000.)
    assert qualify_pair(request,cur,ref,gates)['score']>=9.9


def test_existing_ppo_runner_refuses_unqualified_challenge_before_allocating_world(tmp_path):
    import json
    from driversi_world.world_loop.experiment import run
    path=tmp_path/'config.json';path.write_text(json.dumps({'challenge':{'unqualified':True}}))
    with pytest.raises(ValueError,match='qualified evaluator'):run(path)
