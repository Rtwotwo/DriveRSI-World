"""Render a synchronized three-camera diagnostic, never an exported training dataset."""
import copy
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from driversi_world.runtime.paths import configure_runtime, experiment_path
from .trajectory_video import interpolate_state, edited_path


def prepare(config):
    out = experiment_path(config['output']); work = out/'work'
    configure_runtime(work)
    import cv2
    import torch
    from pyquaternion import Quaternion
    from driversi_world.simulation.envs.base_env import BaseEnv
    from .action_render import _configuration
    from .evidence import scaled_cameras
    from .trajectories import lateral_reference, center_step_action
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.18)
    with Path(config['scenarios']).open('rb') as f: scene = pickle.load(f)[config['scene']]
    scene = copy.deepcopy(scene)
    cameras = config['cameras']; frames = config['frames']
    scene['cameras'] = scaled_cameras(scene['cameras'], cameras, config['width'])
    raw = list(scene['metadata']['openscene_data_infos_dict'].values())[:12]
    real = {}
    for camera in cameras:
        rows = []
        for row in raw:
            p = Path(config['sensor_root'])/row['cams'][camera]['data_path']
            image = cv2.imread(str(p))
            if image is None: raise FileNotFoundError(p)
            rows.append(cv2.cvtColor(cv2.resize(image, (config['width'],config['height']),
                interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
        real[camera] = np.stack(rows)
    spec = dict(config, nuplan_maps=config['maps'])
    env = BaseEnv(_configuration(spec,work), 'ppo-observation-probe', {config['scene']:scene})
    gs = {c:[] for c in cameras}; replay={c:[] for c in cameras}
    geometry={c:[] for c in cameras}; render_times=[]; pose_errors=[]
    try:
        observation=env.reset(seed=0)['render']; agent=env.engine.agent_manager.ego_agent
        reference=lateral_reference(agent.object_track['position'],agent.object_track['heading'],0.)
        anchors=[]
        for i in range(12):
            if i: observation,_,_,_=env.step(center_step_action(agent,reference,i))
            anchors.append(copy.deepcopy(observation['render_state']))
            for c in cameras: replay[c].append(cv2.cvtColor(observation['cameras'][c]['image'],cv2.COLOR_BGR2RGB))
        times=np.arange(frames)/config['fps']; states=[]
        for t in times:
            lower=min(int(t/.5),10); f=(t-lower*.5)/.5
            state=copy.deepcopy(anchors[lower])
            state['agent_state']=interpolate_state(anchors[lower]['agent_state'],anchors[lower+1]['agent_state'],f)
            state['timestamp']=int(anchors[0]['timestamp']+t*1e6); states.append(state)
        ego=np.stack([s['agent_state']['ego'] for s in states])
        xy,yaw,offset=edited_path(ego,times,config['offset_m'])
        renderer=env.engine.render_manager.renderer
        sample_indices=config['geometry_frames']
        for i,state in enumerate(states):
            state['agent_state']['ego'][:2]=xy[i]; state['agent_state']['ego'][-1]=yaw[i]
            renderer.render_depth=i in sample_indices
            torch.cuda.synchronize(); tick=time.perf_counter()
            observed=renderer.render(state)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-tick
            render_times.append(dict(frame=i,seconds=elapsed,depth=renderer.render_depth))
            error=float(np.linalg.norm(observed['ego2global'][:2,3]-xy[i])); pose_errors.append(error)
            if error>1e-4: raise RuntimeError('Camera does not follow requested trajectory')
            for c in cameras:
                gs[c].append(cv2.cvtColor(observed['cameras'][c]['image'],cv2.COLOR_BGR2RGB))
                if renderer.render_depth:
                    # cv2.remap may squeeze a singleton channel: accept HxW and HxWx1.
                    depth = np.asarray(observed['cameras'][c]['depth'])
                    if depth.ndim == 3 and depth.shape[-1] == 1:
                        depth = depth[..., 0]
                    if depth.shape != (config['height'], config['width']):
                        raise ValueError(f'Unexpected rendered depth shape: {depth.shape}')
                    geometry[c].append(depth.astype(np.float32))
            if i%24==0: print('RENDER',i,'seconds',round(elapsed,3),flush=True)
        calibrations={}
        for c in cameras:
            camera=scene['cameras'][c]
            extrinsic=np.eye(4); extrinsic[:3,:3]=Quaternion(camera['sensor2ego_rotation']).rotation_matrix
            extrinsic[:3,3]=camera['sensor2ego_translation']
            calibrations[c]=dict(K=np.asarray(camera['intrinsic']).tolist(),
                distortion=np.asarray(camera['distortion']).tolist(),camera_to_ego=extrinsic.tolist())
            np.savez_compressed(work/f'{c}_input_pending.npz',gs=np.stack(gs[c]),
                depth=np.stack(geometry[c]),original=real[c],replay=np.stack(replay[c]))
            (work/f'{c}_input_pending.npz').replace(work/f'{c}_input.npz')
        record=dict(config=config,calibrations=calibrations,render_times=render_times,
            max_camera_error_m=max(pose_errors),heading_mode='preserve_log_heading',
            trajectory=dict(seconds=times.tolist(),original_xy=ego[:,:2].tolist(),edited_xy=xy.tolist()),
            purpose='PPO observation diagnostics only; no training data export',
            source_original_hz=2,traffic='log replay',training_admitted=False)
        (out/'result.json').write_text(json.dumps(record,indent=2))
        print('PREPARED_THREE_VIEWS',flush=True)
    finally: env.close()


if __name__=='__main__':
    record=json.loads(Path(sys.argv[1]).read_text()); prepare(record.get('config',record))
