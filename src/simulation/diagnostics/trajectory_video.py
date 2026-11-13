"""Prepare real camera frames and dense MTGS renders along an edited camera path.

This is an offline view-synthesis experiment. Traffic follows the recorded log;
the edited trajectory is not a validated vehicle-control or collision rollout.
"""
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from driversi_world.runtime.paths import configure_runtime, experiment_path


def interpolate_state(first, second, fraction):
    """Interpolate common recorded actors; wrap headings across the pi boundary."""
    result = {}
    for name, a in first.items():
        if name not in second:
            result[name] = np.asarray(a).copy()
            continue
        a, b = np.asarray(a), np.asarray(second[name])
        value = a + fraction * (b - a)
        angle = np.arctan2(np.sin(b[-1] - a[-1]), np.cos(b[-1] - a[-1]))
        value[-1] = a[-1] + fraction * angle
        result[name] = value
    return result


def edited_path(ego, times, offset):
    """Quintic lateral camera translation while retaining the recorded heading.

    A near-stationary log has no reliable forward path tangent. This deliberately
    edits the camera path and does not claim a dynamically executable lane change.
    """
    if not 0 < abs(offset) <= 3.5:
        raise ValueError('This visual demo supports a bounded lateral shift up to 3.5 m')
    phase = (times - times[0]) / (times[-1] - times[0])
    shift = offset * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    normal = np.column_stack((-np.sin(ego[:, -1]), np.cos(ego[:, -1])))
    position = ego[:, :2] + shift[:, None] * normal
    yaw = ego[:, -1].copy()
    return position, yaw, shift


def run(config):
    out = experiment_path(config['output'])
    work = out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    configure_runtime(work)
    import cv2
    import torch
    from driversi_world.simulation.envs.base_env import BaseEnv
    from driversi_world.simulation.diagnostics.action_render import _configuration
    from driversi_world.simulation.diagnostics.evidence import scaled_cameras
    from driversi_world.simulation.diagnostics.trajectories import lateral_reference, center_step_action

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.18)
    scenes = pickle.load(Path(config['scenarios']).open('rb'))
    original = scenes[config['scene']]
    raw = list(original['metadata']['openscene_data_infos_dict'].values())[:12]
    stamps = np.array([frame['timestamp'] for frame in raw], dtype=np.int64)
    times = np.arange(config['frames']) / config['fps']
    source_times = (stamps - stamps[0]) / 1e6
    if times[-1] > source_times[-1]:
        raise ValueError('The requested clip exceeds the available original log')
    raw_files = [Path(config['sensor_root']) / row['cams']['CAM_F0']['data_path'] for row in raw]
    if not all(path.is_file() for path in raw_files):
        raise FileNotFoundError('Original camera frames are required; GS is not a substitute')
    real = []
    for path in raw_files:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Unreadable log frame: {path}')
        frame = cv2.resize(frame, (config['width'], config['height']), interpolation=cv2.INTER_AREA)
        real.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    indices = np.argmin(abs(times[:, None] - source_times[None, :]), axis=1)
    real = np.stack(real)[indices]

    scene = copy.deepcopy(original)
    scene['cameras'] = scaled_cameras(scene['cameras'], ['CAM_F0'], config['width'])
    spec = dict(config, nuplan_maps=config['maps'], asset_root=config['asset_root'])
    env = BaseEnv(_configuration(spec, work), 'trajectory-video', {config['scene']: scene})
    try:
        initial = env.reset(seed=0)
        if initial is None:
            raise RuntimeError('Scene initialization failed')
        observation = initial['render']
        agent = env.engine.agent_manager.ego_agent
        log_reference = lateral_reference(agent.object_track['position'], agent.object_track['heading'], 0.)
        anchors = []
        for i in range(12):
            if i:
                observation, _, _, _ = env.step(center_step_action(agent, log_reference, i))
            anchors.append(copy.deepcopy(observation['render_state']))
        dense = []
        for seconds in times:
            lower = min(int(seconds / .5), len(anchors) - 2)
            weight = (seconds - lower * .5) / .5
            state = copy.deepcopy(anchors[lower])
            state['agent_state'] = interpolate_state(anchors[lower]['agent_state'],
                                                     anchors[lower+1]['agent_state'], weight)
            state['timestamp'] = int(anchors[0]['timestamp'] + seconds * 1e6)
            dense.append(state)
        ego = np.stack([state['agent_state']['ego'] for state in dense])
        position, yaw, shift = edited_path(ego, times, config['offset_m'])
        gs, checks = [], []
        renderer = env.engine.render_manager.renderer
        for i, state in enumerate(dense):
            state['agent_state']['ego'][:2] = position[i]
            state['agent_state']['ego'][-1] = yaw[i]
            rendered = renderer.render(state)
            error = float(np.linalg.norm(rendered['ego2global'][:2, 3] - position[i]))
            if error > 1e-4:
                raise AssertionError(f'Rendered camera did not follow the path: {error}')
            rgb = cv2.cvtColor(rendered['cameras']['CAM_F0']['image'], cv2.COLOR_BGR2RGB)
            if rgb.shape != (config['height'], config['width'], 3) or rgb.std() < 1:
                raise ValueError('Invalid GS frame')
            gs.append(rgb)
            checks.append(error)
            if i % 24 == 0:
                print('GS_FRAME', i, 'offset_m', round(float(shift[i]), 3), flush=True)
        gs = np.stack(gs)
        # One compact intermediate file replaces hundreds of individual PNGs.
        np.savez_compressed(work / 'inputs.npz', original=real, gs=gs)
        record = dict(config=config, scene=config['scene'], camera='CAM_F0', source_camera_hz=2,
            display_fps=config['fps'], frames=config['frames'], duration_seconds=times[-1],
            resampling='Nearest original log frame; no synthetic interpolation on the original side',
            source_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in raw_files],
            source_times_seconds=source_times.tolist(), source_frame_indices=indices.tolist(),
            trajectory=dict(heading_mode='preserve_log_heading',
                times_seconds=times.tolist(), original_xy=ego[:, :2].tolist(),
                edited_xy=position.tolist(), original_yaw=ego[:, -1].tolist(), edited_yaw=yaw.tolist(),
                lateral_offset_m=shift.tolist(), max_camera_position_error_m=max(checks)),
            traffic='Recorded actor states, interpolated in time; not reactive',
            training_admitted=False)
        (out / 'result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        for name, array in [('original', real), ('new_path', gs)]:
            cv2.imwrite(str(work / (name + '_last.jpg')), cv2.cvtColor(array[-1], cv2.COLOR_RGB2BGR))
        print('PREPARED', config['frames'], 'frames', 'offset_m', config['offset_m'], flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    document = json.loads(Path(sys.argv[1]).read_text())
    run(document.get('config', document))
