"""MTGS environment adapter with executed dynamics, reactive traffic and rewards.

The old data-export and candidate-scoring managers are disabled. The wrapper
returns one actual transition and preserves proper termination versus truncation.
"""
import copy
import pickle
import time
import hashlib
from pathlib import Path
import numpy as np

from driversi_world.runtime.paths import configure_runtime, experiment_path
from driversi_world.simulation.diagnostics.action_render import _configuration
from driversi_world.simulation.diagnostics.evidence import scaled_cameras
from .dynamics import TrajectoryController, rotate
from .traffic import ReactiveIDMPolicy, PathStateController
from .rewards import GeometryReward, REWARD_VERSION
from .challenges import (catalogue, validate as validate_challenge, digest as challenge_digest,
                         apply_to_policies, CHALLENGE_ENVIRONMENT)

ENVIRONMENT_VERSION = 'mtgs-reactive-si-v1'
ACTION_VERSION = 'rear-trajectory-bicycle-05s-v1'
CAMERAS = ['CAM_L0','CAM_F0','CAM_R0']


class ClosedLoopWorld:
    def __init__(self, config):
        import torch
        from driversi_world.simulation.envs.base_env import BaseEnv
        self.config = config
        self.challenge = copy.deepcopy(config.get('challenge'))
        # Structural requests are admitted for candidate evaluation only. A future
        # training adapter must independently qualify paired reference evidence.
        if self.challenge is not None and config.get('challenge_phase') != 'evaluation':
            raise ValueError('Unqualified challenge is only allowed in evaluation phase')
        self.work = experiment_path(config['output'])/'work'
        configure_runtime(self.work)
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.18)
        with Path(config['scenarios']).open('rb') as handle:
            self.scene = copy.deepcopy(pickle.load(handle)[config['scene']])
        self.scene['cameras'] = scaled_cameras(self.scene['cameras'],CAMERAS,1920)
        self.raw = list(self.scene['metadata']['openscene_data_infos_dict'].values())
        spec = dict(config,nuplan_maps=config['maps'])
        settings = _configuration(spec,self.work)
        settings.ego_controller = 'two_stage_controller'
        settings.agent_policy = 'idm_policy'
        settings.agent_navigation = 'idm_navigation'
        settings.agent_controller = 'log_play_controller'  # replaced before any action is executed
        settings.max_step = 11
        self.env = BaseEnv(settings,'world-ppo', {config['scene']:self.scene})
        self.last = None
        self.episode = 0
        self.executed_environment_steps = 0
        self.done = True

    @property
    def ego(self):
        return self.env.engine.agent_manager.ego_agent

    @property
    def step_index(self):
        return self.env.engine.episode_step

    def _install(self):
        manager = self.env.engine.agent_manager
        for key, actor in manager.get_dynamic_agents.items():
            if key == 'ego':
                if not isinstance(actor.controller,TrajectoryController):
                    actor.controller = TrajectoryController(actor)
            elif not isinstance(actor.policy,ReactiveIDMPolicy):
                actor.policy = ReactiveIDMPolicy(actor)
                actor.controller = PathStateController(actor)

    def participants(self):
        result = {}
        manager = self.env.engine.agent_manager
        for key,a in manager.all_agents.items():
            result[key] = dict(position=np.asarray(a.current_position).copy(),heading=float(a.current_heading),
                               speed=float(a.current_speed),length=float(a.length),width=float(a.width))
        # The existing agent manager omits non-vehicle participants. Preserve their
        # logged collision boxes explicitly; they are not claimed to be reactive.
        for key,state in self.logged_nonvehicles.items():
            i = self.step_index
            if i >= len(state['valid']) or not bool(np.asarray(state['valid'][i]).item()):
                continue
            result[key] = dict(position=np.asarray(state['position'][i])[:2],
                heading=float(np.asarray(state['heading'][i]).item()),speed=float(np.linalg.norm(state['velocity'][i])),
                length=float(np.asarray(state['length'][i]).item()),width=float(np.asarray(state['width'][i]).item()))
        return result

    def reset(self):
        result = self.env.reset(seed=0)
        if result is None:
            raise RuntimeError('Scenario reset returned no observation')
        self.episode += 1
        self.last = result['render']
        self._install()
        manager = self.env.engine.agent_manager
        if self.challenge is not None:
            capability = catalogue(self.scene,
                self.env.engine.render_manager.renderer.asset_manager.background_asset,
                hashlib.sha256(Path(self.config['scenarios']).read_bytes()).hexdigest())
            validate_challenge(self.challenge,capability)
            actor_id = self.challenge['edit']['actor_id']
            if actor_id not in manager.get_dynamic_agents:
                raise ValueError('Challenge actor is not an active dynamic simulation object')
        self.logged_nonvehicles = {key:manager._process_agent_track(key,track)[0]
            for key,track in self.scene['object_track'].items() if track['type'] in ('PEDESTRIAN','CYCLIST')}
        self.reward_model = GeometryReward(self.scene,self.ego)
        self.done = False
        return self.last

    def trajectory(self, ego_trajectory):
        """Policy's eight rear-axle ego poses -> native nine center poses."""
        from driversi_world.simulation.common.dataclasses import Trajectory
        poses = np.asarray(ego_trajectory,dtype=float)
        if poses.shape != (8,3) or not np.isfinite(poses).all():
            raise ValueError('Expected a finite 8x3 native policy trajectory')
        state = self.ego.controller.state
        rear = np.vstack((state[:2],poses[:,:2] @ rotate(state[2]).T + state[:2]))
        heading = np.r_[state[2],poses[:,2]+state[2]]
        center = rear + self.ego.vehicle.rear_axle_to_center*np.column_stack((np.cos(heading),np.sin(heading)))
        velocity = np.gradient(center,.5,axis=0)
        return Trajectory(waypoints=center,velocities=velocity,headings=heading,
                          angular_velocities=np.gradient(np.unwrap(heading),.5))

    def step(self, ego_trajectory):
        import torch
        if self.done:
            raise RuntimeError('Reset the completed episode before stepping')
        if self.executed_environment_steps >= self.config.get('environment_step_budget',512):
            raise RuntimeError('Declared actual environment-step budget exhausted')
        self._install()
        challenge_edit = apply_to_policies(self.env.engine.agent_manager.get_dynamic_agents,
                                          self.challenge,self.step_index)
        self.env.engine.closed_loop_snapshot = self.participants()
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.last,_,_,_ = self.env.step(self.trajectory(ego_trajectory))
        self.executed_environment_steps += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        reward,terms,metrics,terminated = self.reward_model.evaluate(
            self.ego,self.participants(),self.ego.controller.control)
        truncated = bool(self.step_index>=11 and not terminated)
        self.done = terminated or truncated
        camera_error = float(np.linalg.norm(self.last['ego2global'][:2,3] -
            (self.ego.controller.state[:2]+self.env.engine.render_manager.local2global_translation_xy)))
        if camera_error>1e-4:
            raise RuntimeError('Rendered camera does not follow the executed ego pose')
        traffic = {key:dict(copy.deepcopy(a.policy.last_decision),position=np.asarray(a.current_position).tolist())
            for key,a in self.env.engine.agent_manager.get_dynamic_agents.items()
            if isinstance(a.policy,ReactiveIDMPolicy)}
        return self.last,reward,terminated,truncated,dict(
            environment_version=CHALLENGE_ENVIRONMENT if self.challenge else ENVIRONMENT_VERSION,reward_version=REWARD_VERSION,
            action_transform_version=ACTION_VERSION,control=self.ego.controller.control.copy(),
            challenge=None if self.challenge is None else dict(
                request_sha256=challenge_digest(self.challenge),actual_target_edit=challenge_edit,
                phase='evaluation',training_ready=False),
            reward_terms=terms,metrics=metrics,state=self.ego.controller.state.tolist(),
            camera_error_m=camera_error,step_seconds=elapsed,traffic=traffic,
            images_sha256={c:hashlib.sha256(self.last['cameras'][c]['image'].tobytes()).hexdigest() for c in CAMERAS})

    def straight_action(self, speed=1.0, lateral=0.0):
        t = np.arange(1,9)*.5
        return np.column_stack((speed*t,lateral*(1-np.exp(-t)),np.zeros(8)))

    def close(self):
        self.env.close()

