"""CPU-only World capability/receipt CLI; emits one JSON record on stdout.

Run from this project's exp directory. --config selects existing managed scene
assets. --request validates one challenge, while --smoke executes real IDM scalar
dynamics on scene-derived initial speeds. The smoke is not a camera/traffic rollout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

from driversi_world.simulation.closed_loop.challenges import catalogue,receipt,target_speed
from .challenger import candidate_requests


def load_capability(config):
    """Read existing assets on CPU without importing anything from clones or Policy."""
    import torch
    torch.set_num_threads(2)
    scene_path = Path(config['scenarios']).resolve(strict=True)
    scene = pickle.loads(scene_path.read_bytes())[config['scene']]
    asset_id = scene['metadata']['digitaltwin_asset_id']
    asset_path = Path(config['asset_root'])/'assets'/asset_id/'background'/f'{asset_id}.ckpt'
    asset = torch.load(asset_path,map_location='cpu',weights_only=False)
    cap = catalogue(scene,asset,hashlib.sha256(scene_path.read_bytes()).hexdigest())
    provenance = dict(scene_path=str(scene_path),asset_path=str(asset_path),
        asset_sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        split_evidence='Asset root is navtrain; independent training manifest still required')
    return cap,provenance


def smoke(capability,requests):
    """Free-road kinematic sensitivity only: no lane, leader, renderer or ego policy."""
    from driversi_world.simulation.closed_loop.traffic import idm_acceleration,travel
    trials = []
    for request in requests:
        structural = receipt(request,capability)
        actor_id = request['edit']['actor_id']
        actor = capability['actors'][actor_id]
        branches = {}
        for label,edit in [('baseline',None),('candidate',request)]:
            speed,distance,rows = actor['initial_speed_mps'],0.,[]
            for step in range(11):
                desired = target_speed(actor['nominal_desired_speed_mps'],edit,actor_id,step)
                acceleration = idm_acceleration(speed,desired)
                advance,speed = travel(speed,acceleration,.5)
                distance += advance
                if not -4. <= acceleration <= 2. or speed < 0 or advance < 0:
                    raise RuntimeError('SI kinematic bounds violated')
                rows.append(dict(step=step,speed_mps=speed,distance_m=distance,
                                 acceleration_mps2=acceleration,target_mps=desired))
            branches[label] = rows
        trials.append(dict(request=request,receipt=structural,
            final_distance_difference_m=branches['candidate'][-1]['distance_m']-branches['baseline'][-1]['distance_m'],
            branches=branches))
    return dict(kind='scene_seeded_free_road_idm_cpu_smoke',trials=trials,
        training_ready=False,full_traffic_simulation=False,rendered=False,
        interpretation='Sensitivity and bounds only; not difficulty, solvability or PPO evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    from driversi_world.runtime.paths import experiment_path
    experiment_path(Path.cwd())
    if args.request and args.smoke:
        parser.error('Select one receipt or the bounded CPU smoke')
    started=time.perf_counter()
    config=json.loads(args.config.read_text())
    cap,provenance=load_capability(config)
    record=dict(schema='driversi.world-challenge-cpu.v1',status='complete',
                capability=cap,provenance=provenance,gpu_used=False)
    if args.request:
        record['receipt']=receipt(json.loads(args.request.read_text()),cap)
    if args.smoke:
        record['smoke']=smoke(cap,candidate_requests(cap,seed=16116,limit=8))
    record['seconds']=time.perf_counter()-started
    print(json.dumps(record,indent=2,allow_nan=False))


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=''
    main()
