"""Run a real, bounded action-to-MTGS integration check on qualified log data.

Usage: python -B -m driversi_world.simulation.diagnostics.action_render SPEC.json
The JSON specification records input paths and output limits. This diagnostic
does not train a policy, assess safety, or admit generated data for training.
"""
import copy
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

from driversi_world.runtime.paths import configure_runtime, experiment_path


def _configuration(spec, output):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / 'configs'), version_base='1.2'):
        cfg = compose(config_name='default_runner')
    settings = dict(
        output_dir=str(output), data_output_dir=str(output / 'unused-export'),
        completed_scenarios_dir=str(output / 'completed'),
        data_file_path=spec['scenarios'], asset_folder_path=str(Path(spec['asset_root']) / 'assets'),
        nuplan_map_root=spec['nuplan_maps'], use_planner_actions=False,
        ego_client='base_client', ego_policy='env_input_policy',
        ego_navigation='trajectory_navigation', ego_controller='log_play_controller',
        agent_policy='trajectory_policy', agent_navigation='trajectory_navigation',
        agent_controller='log_play_controller', with_data_manager=False,
        with_metric_manager=False, with_dense_reward_manager=False,
        save_data=False, enable_resume=False, max_step=11,
        number_of_cpus_allocated_per_simulation=2, seed=0,
    )
    for key, value in settings.items():
        cfg[key] = value
    return cfg


def _snapshot(agent, observation, step, origin):
    import numpy as np
    expected_xy = np.asarray(agent.rear_vehicle.current_position) + origin
    actual_xy = observation['ego2global'][:2, 3]
    return dict(step=step, seconds=step * .5, center_xy=agent.current_position.tolist(),
                rear_axle_global_xy=expected_xy.tolist(), heading=float(agent.current_heading),
                speed_mps=float(agent.current_speed), camera_ego2global=observation['ego2global'].tolist(),
                camera_xy_error_m=float(np.linalg.norm(actual_xy - expected_xy)),
                image_sha256={name: hashlib.sha256(cam['image'].tobytes()).hexdigest()
                              for name, cam in observation['cameras'].items()})


def _repeat_probe(renderer, observation):
    """Exercise the real GPU path twice using the exact same caller-owned input."""
    import numpy as np
    from driversi_world.simulation.diagnostics.evidence import image_difference
    state = observation['render_state']
    before = copy.deepcopy(state['agent_state'])
    repeated = renderer.render(state)
    preserved = all(np.array_equal(value, state['agent_state'][name]) for name, value in before.items())
    differences = {name: image_difference(cam['image'], repeated['cameras'][name]['image'])
                   for name, cam in observation['cameras'].items()}
    return {'input_preserved': preserved, 'image_differences': differences,
            'pose_max_difference': float(np.max(np.abs(observation['ego2global'] - repeated['ego2global'])))}


def run(spec):
    output = experiment_path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    configure_runtime(output)
    os.environ['MAX_JOBS'] = '2'  # Bound optional CUDA-extension compilation as well.
    started = time.monotonic()
    report = dict(schema='driversi.action-render-diagnostic.v1', status='running',
                  trainable=False, parameter_updates=0, seed=0,
                  scope='Scripted log-future intervention with ideal centered trajectory tracking',
                  traffic='log_replay', geometry='fixed_pretrained_mtgs',
                  limitations=['No autonomous policy or PPO update', 'No reactive traffic',
                               'No independent novel-view ground truth', 'No safety or collision admission'],
                  spec=spec, branches={})
    (output / 'spec.json').write_text(json.dumps(spec, indent=2))
    env = None
    try:
        import numpy as np
        import torch
        from driversi_world.data.selection import qualify, inspect_scenarios
        from driversi_world.simulation.envs.base_env import BaseEnv
        from driversi_world.simulation.diagnostics.trajectories import lateral_reference, center_step_action
        from driversi_world.simulation.diagnostics.evidence import (
            scaled_cameras, image_difference, write_image, write_comparison,
        )
        if not 0 < spec.get('gpu_memory_fraction', .18) <= .18:
            raise ValueError('Shared-GPU pilot memory limit must be in (0, 0.18]')
        if spec.get('width', 480) not in (320, 480, 640):
            raise ValueError('Use one of the bounded diagnostic resolutions')
        torch.set_num_threads(2)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.set_per_process_memory_fraction(spec.get('gpu_memory_fraction', .18))
        selection = json.loads(Path(spec['selection']).read_text())
        qualify(selection)
        eligibility = inspect_scenarios(spec['scenarios'], selection)
        report['source_checks'] = eligibility
        with Path(spec['scenarios']).open('rb') as handle:
            scenes = pickle.load(handle)
        # Visual diagnostics retain the production admission decision. Pedestrians
        # may appear in pretrained imagery but are not simulated collision agents.
        # Never relabel this path as a qualified training rollout.
        token = spec['token']
        source = next(row for row in selection['records'] if row['token'] == token)
        scene_name = source['log_name'] + '-' + token
        original = scenes[scene_name]
        frames = list(original['metadata']['openscene_data_infos_dict'].values())
        stamps = [int(frame['timestamp']) for frame in frames]
        if not (original['sample_rate'] == 10 and len(frames) >= 12
                and frames[3]['token'] == token
                and all(frame['log_name'] == source['log_name'] for frame in frames)
                and all(450000 <= b - a <= 550000 for a, b in zip(stamps, stamps[1:]))):
            raise ValueError('Incomplete or misaligned visual diagnostic window')
        unsupported = [key for key, track in original['object_track'].items()
                       if track['type'] in ('PEDESTRIAN', 'CYCLIST')
                       and np.any(track['state']['valid'][:12])]
        report['unsupported_collision_participants'] = unsupported
        report['production_admission_passed'] = token in eligibility['completed']
        if unsupported:
            report['limitations'].append('Pedestrians/cyclists are absent from collision-state simulation')
        report['scene'] = scene_name
        report['scenarios_sha256'] = hashlib.sha256(Path(spec['scenarios']).read_bytes()).hexdigest()
        cameras = spec.get('cameras', ['CAM_L0', 'CAM_F0', 'CAM_R0'])
        if not cameras or len(cameras) > 3 or len(set(cameras)) != len(cameras):
            raise ValueError('Select one to three distinct cameras')
        all_images = {}
        branch_specs = [('replay', 0.), ('left_0.5m', .5), ('right_0.5m', -.5)]
        for branch, offset in branch_specs:
            scene = copy.deepcopy(original)
            scene['cameras'] = scaled_cameras(scene['cameras'], cameras, spec.get('width', 480))
            folder = output / branch
            folder.mkdir()
            env = BaseEnv(_configuration(spec, folder), branch, {scene_name: scene})
            initial = env.reset(seed=0)
            if initial is None:
                raise RuntimeError('Environment did not initialize a scene')
            observation = initial['render']
            agent = env.engine.agent_manager.ego_agent
            origin = -np.array(scene['metadata']['old_origin_in_current_coordinate'])
            reference = lateral_reference(agent.object_track['position'], agent.object_track['heading'], offset)
            records, images, step_seconds = [], [], []
            for step in range(12):
                if step:
                    action = center_step_action(agent, reference, step)
                    torch.cuda.synchronize()
                    step_started = time.monotonic()
                    observation, _, _, truncated = env.step(action)
                    torch.cuda.synchronize()
                    step_seconds.append(time.monotonic() - step_started)
                    if truncated != (step == 11):
                        raise RuntimeError('Unexpected history/future termination boundary')
                records.append(_snapshot(agent, observation, step, origin))
                frame_images = {name: observation['cameras'][name]['image'].copy() for name in cameras}
                if any(image.std() < 1 for image in frame_images.values()):
                    raise RuntimeError('Blank or nearly uniform real-scene render')
                images.append(frame_images)
                for name, image in frame_images.items():
                    write_image(folder / f'{step:02d}_{name}.png', image)
                if branch == 'replay' and step == 3:
                    report['repeat_render'] = _repeat_probe(env.engine.render_manager.renderer, observation)
                print(json.dumps({'branch': branch, 'frame': step,
                                  'camera_xy_error_m': records[-1]['camera_xy_error_m']}), flush=True)
            report['branches'][branch] = {'offset_m': offset, 'frames': records,
                                          'step_wall_seconds': step_seconds}
            all_images[branch] = images
            env.close()
            env = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        comparisons = {}
        baseline = report['branches']['replay']['frames']
        for branch, _ in branch_specs[1:]:
            frames = report['branches'][branch]['frames']
            displacement = [float(np.linalg.norm(np.array(a['center_xy']) - b['center_xy']))
                            for a, b in zip(frames, baseline)]
            pixel_differences = [{name: image_difference(a[name], b[name]) for name in cameras}
                                 for a, b in zip(all_images[branch], all_images['replay'])]
            comparisons[branch] = dict(center_displacement_m=displacement,
                                       image_differences=pixel_differences)
        report['comparisons'] = comparisons
        checks = {
            'same_initial_history': all(np.max(c['center_displacement_m'][:4]) < 1e-6 for c in comparisons.values()),
            'bounded_0_5m_intervention': all(.49 < c['center_displacement_m'][-1] < .501 and
                                           max(c['center_displacement_m']) < .501 for c in comparisons.values()),
            'camera_tracks_actual_state': all(f['camera_xy_error_m'] < 1e-4
                                             for b in report['branches'].values() for f in b['frames']),
            'action_changes_observation': all(any(d['mae_0_255'] > .01 for d in c['image_differences'][-1].values())
                                             for c in comparisons.values()),
            'repeat_preserves_input': report['repeat_render']['input_preserved'],
            'repeat_preserves_image': all(d['mae_0_255'] == 0 for d in report['repeat_render']['image_differences'].values()),
        }
        report['checks'] = checks
        report['artifacts'] = write_comparison(output, all_images, cameras)
        report['peak_torch_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['status'] = 'completed' if all(checks.values()) else 'failed_checks'
        if not all(checks.values()):
            raise RuntimeError('Diagnostic assertion failed: ' + str(checks))
        return report
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if env is not None:
            env.close()
        report['wall_seconds'] = time.monotonic() - started
        (output / 'result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    result = run(json.loads(Path(sys.argv[1]).read_text()))
    print('ACTION_RENDER_WITNESS', result['status'], result['checks'], flush=True)
