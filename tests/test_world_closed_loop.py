"""Analytical checks for physical direction, SI units and actual-state rewards."""
from types import SimpleNamespace
import numpy as np
import pytest
from driversi_world.simulation.closed_loop.dynamics import bicycle_step
from driversi_world.simulation.closed_loop.traffic import idm_acceleration, travel
from driversi_world.simulation.closed_loop.rewards import GeometryReward
from driversi_world.world_loop.observations import inertial_state


def test_bicycle_stop_never_reverses_and_left_steer_moves_left():
    state = np.array([0.,0.,0.,.2,0.])
    stopped, _ = bicycle_step(state,np.zeros((9,2)))
    assert 0 <= stopped[3] < state[3] and stopped[0] >= 0
    path = np.column_stack((np.arange(9)*.5,np.arange(9)*.15))
    left, _ = bicycle_step(np.array([0.,0.,0.,1.,0.]),path)
    assert left[1]>0 and left[2]>0


def test_idm_si_formula_and_stopping_distance():
    # v=10, v0=20, lead v=10, gap=17 => desired gap is exactly 17 m.
    assert idm_acceleration(10,20,17,10)==pytest.approx(-.125)
    assert idm_acceleration(10,20,5,0)<idm_acceleration(10,20,50,10)
    distance,speed=travel(1,-4,.5)
    assert distance==pytest.approx(.125) and speed==0


def test_turning_observation_contains_centripetal_acceleration():
    # v=2 m/s on a 10 m radius circle has ay=0.4 m/s².
    state=np.array([0.,0.,0.,2.,np.arctan(3.089/10)])
    assert inertial_state(state,{'acceleration':-.5},3.089)==pytest.approx([2.,0.,-.5,.4])


def test_reward_changes_with_executed_pose_and_collision():
    scene={'map_features':{'lane':{'type':'LANE_SURFACE_STREET','polygon':[[-10,-2],[30,-2],[30,2],[-10,2]]}}}
    ego=SimpleNamespace(current_position=np.array([0.,0.]),current_heading=0.,length=4.,width=2.,
                        object_track={'position':np.array([[0.,0.],[10.,0.]])})
    reward=GeometryReward(scene,ego)
    control={'acceleration':0.,'steering_rate':0.}
    ego.current_position=np.array([1.,0.])
    value,_,metrics,done=reward.evaluate(ego,{},control)
    assert value==pytest.approx(.2) and not done and metrics['delta_progress_m']==pytest.approx(1)
    other={'npc':dict(position=[2.,0.],heading=0.,length=4.,width=2.)}
    value,_,metrics,done=reward.evaluate(ego,other,control)
    assert done and metrics['collisions']==['npc'] and value<=-10
