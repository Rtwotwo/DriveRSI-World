"""Bounded traffic edits with explicit structural versus rollout qualification.

The first interface changes one existing vehicle's IDM target speed. It cannot
teleport, spawn, resize, change lanes, alter rewards, or certify solvability.
Step k applies to the executed interval [k*0.5, (k+1)*0.5) seconds.
"""
import copy
import hashlib
import json
import math
import re
import numpy as np
from .rewards import footprint

SCHEMA = 'driversi.world-challenge.v1'
VERSION = 'mtgs-idm-target-speed-v1'
CHALLENGE_ENVIRONMENT = 'mtgs-reactive-si-challenge-v1'
HORIZON = 11


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def catalogue(scene, asset, scene_sha256):
    """Intersect always-present dynamic vehicles with dynamic rigid GS identities.

    Uses the engine's current motion threshold (3 m coordinate standard deviation).
    No logged future trajectory is exposed to the driving policy by this API.
    """
    renderable = set()
    for name,item in asset.items():
        state = item.get('state_dict', {})
        trans = state.get('instance_trans')
        if (item.get('config',{}).get('type') in ('RigidSubModel','MirroredRigidSubModel')
                and trans is not None and trans.squeeze().ndim == 2):
            renderable.add(name.rsplit('_',1)[-1])
    actors = {}
    boxes = {}
    for key,item in scene['object_track'].items():
        state = item['state']
        valid = np.asarray(state['valid']).reshape(-1) == 1
        if not len(valid) or not valid[0]:
            continue
        length,width = (float(np.asarray(state[k]).reshape(-1)[0]) for k in ('length','width'))
        position = np.asarray(state['position'])[0,:2]
        heading = float(np.asarray(state['heading']).reshape(-1)[0])
        if length > 0 and width > 0 and np.isfinite([*position,heading,length,width]).all():
            boxes[key] = footprint(position, heading, length, width)
        if key == 'ego' or item['type'] != 'VEHICLE' or key not in renderable:
            continue
        if len(valid) <= HORIZON or not valid[:HORIZON+1].all():
            continue
        points = np.asarray(state['position'])[valid,:2]
        if not np.isfinite(points).all() or np.max(np.std(points,axis=0)) <= 3:
            continue
        speed = float(np.linalg.norm(np.asarray(state['velocity'])[0,:2]))
        if key not in boxes or not math.isfinite(speed):
            continue
        actors[key] = dict(initial_speed_mps=speed, initial_position=position.tolist(),
            nominal_desired_speed_mps=max(4.,min(speed,9.7)), length_m=length,width_m=width,
            presence='all_steps_0_to_11', dynamic_rigid_gs=True)
    for key,actor in actors.items():
        actor['initial_overlap_with'] = sorted(k for k,b in boxes.items()
            if k != key and boxes[key].intersection(b).area > .01)
    value = dict(schema='driversi.world-challenge-capability.v1', version=VERSION,
        scene_id=scene['id'], scene_sha256=scene_sha256, backend='mtgs_reactive_idm',
        dt_seconds=.5, horizon_steps=HORIZON, camera_ids=['CAM_L0','CAM_F0','CAM_R0'],
        edit_bounds=dict(desired_speed_scale=[.5,1.5],start_step=[3,10],duration_steps=[1,8]),
        actors=actors, split_validation='requires_training_manifest',
        unsupported=['spawn','teleport','vehicle_replacement','lane_change','signal_edit',
                     'weather','geometry_generation','policy_training_in_cpu_validator'],
        qualification='structural_only; edited rendering and paired reference rollout required')
    return dict(value, capability_sha256=digest(value))


def validate(request, capability):
    """Reject unknown fields and stale identities rather than silently ignoring edits."""
    keys = {'schema','request_id','scene_id','scene_sha256','capability_sha256',
            'split','seed','objective','edit'}
    if set(request) != keys or request['schema'] != SCHEMA:
        raise ValueError('Exact driversi.world-challenge.v1 fields required')
    unhashed = {k:v for k,v in capability.items() if k != 'capability_sha256'}
    if digest(unhashed) != capability['capability_sha256']:
        raise ValueError('Corrupt capability record')
    for field in ('scene_id','scene_sha256','capability_sha256'):
        if request[field] != capability[field]:
            raise ValueError('Stale or foreign '+field)
    if not isinstance(request['request_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',request['request_id']):
        raise ValueError('Invalid request_id')
    if request['split'] != 'train' or request['objective'] != 'reference_regret':
        raise ValueError('Only train-declared reference_regret requests supported')
    if type(request['seed']) is not int or not 0 <= request['seed'] < 2**32:
        raise ValueError('Invalid seed')
    edit = request['edit']
    if not isinstance(edit,dict) or set(edit) != {'actor_id','desired_speed_scale','start_step','duration_steps'}:
        raise ValueError('Unsupported edit fields')
    actor = capability['actors'].get(edit['actor_id'])
    if actor is None:
        raise ValueError('Actor must be present, dynamic, non-ego and represented by rigid GS')
    if actor['initial_overlap_with']:
        raise ValueError('Initial actor overlap')
    scale,start,duration = (edit[k] for k in ('desired_speed_scale','start_step','duration_steps'))
    if type(scale) not in (float,int) or not math.isfinite(scale) or not .5 <= scale <= 1.5:
        raise ValueError('Target speed scale outside [.5,1.5]')
    if (type(start) is not int or type(duration) is not int or not 3 <= start <= 10
            or not 1 <= duration <= 8 or start+duration > HORIZON):
        raise ValueError('Edit must start after warmup and end within the episode')
    return copy.deepcopy(request)


def receipt(request, capability):
    """A structural receipt deliberately cannot authorize training or claim difficulty."""
    validate(request,capability)
    return dict(schema='driversi.world-challenge-receipt.v1', request_id=request['request_id'],
        request_sha256=digest(request), capability_sha256=capability['capability_sha256'],
        capability_version=VERSION, status='structurally_valid', edit=copy.deepcopy(request['edit']),
        actual_edit_result=None, reference_validation='pending', rendering_validation='pending',
        split_validation='pending_training_manifest', training_ready=False,
        failure_similarity='unverified', interpretation='Target-speed edit only; no executed rollout yet')


def target_speed(nominal, request, actor_id, step):
    """Stateless schedule: ending an edit restores the target, never teleports velocity."""
    if request is None:
        return nominal
    edit = request['edit']
    active = (actor_id == edit['actor_id'] and edit['start_step'] <= step
              < edit['start_step']+edit['duration_steps'])
    return nominal * edit['desired_speed_scale'] if active else nominal


def apply_to_policies(agents, request, step):
    """Called once before the shared traffic snapshot, so all actors see one schedule."""
    changed = {}
    for key,actor in agents.items():
        policy = actor.policy
        if not hasattr(policy,'nominal_desired_speed'):
            continue
        policy.desired_speed = target_speed(policy.nominal_desired_speed, request, key, step)
        if request is not None and key == request['edit']['actor_id']:
            changed[key] = dict(step=step, nominal_desired_speed_mps=policy.nominal_desired_speed,
                               desired_speed_mps=policy.desired_speed)
    return changed
