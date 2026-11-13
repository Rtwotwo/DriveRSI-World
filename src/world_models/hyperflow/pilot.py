"""Run one bounded first-frame / endpoint-conditioned driving-video proposal.

Launch from exp with ``python -m driversi_world.world_models.hyperflow.pilot``.
The output is an offline visual proposal. It cannot replace the simulator's
state, collision geometry, traffic response or reward in RSI/PPO.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from driversi_world.runtime.paths import configure_runtime, experiment_path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    case = config['cases'][args.case]
    out = experiment_path(Path(config['output_root']) / args.case)
    out.mkdir(parents=True, exist_ok=True)
    configure_runtime(out)

    import av
    import numpy as np
    import torch
    from PIL import Image
    from .evaluation import compare_clip
    from .loader import load_pipeline

    torch.set_num_threads(config['cpu_threads'])
    frames, width, height = config['num_frames'], config['width'], config['height']
    if frames < 124 or frames > 362 or (frames - 5) % 17:
        raise ValueError('Frame count must be native 17*n+5 in [124,362]')
    if width % 32 or height % 32:
        raise ValueError('Canvas must be divisible by 32')
    # This pilot is restricted to physical files in project weights/exp.
    root = Path(__file__).resolve().parents[3]
    for name in ('base_model', 'weights'):
        if not Path(config[name]).resolve().is_relative_to(root / 'weights'):
            raise ValueError(f'{name} must be under project weights')
    references = [experiment_path(path) for path in case['references']]
    first = experiment_path(case['first_image'])
    last = experiment_path(case['last_image']) if case.get('last_image') else None
    # Keep immutable input evidence and exact bytes alongside each attempt.
    evidence = dict(case=args.case, config=config, source_camera='CAM_F0',
        source_scene=config['scene'], first_image_sha256=file_hash(first),
        last_image_sha256=file_hash(last) if last else None,
        source_span_seconds=config['source_span_seconds'],
        generated_frame_span_seconds=(frames - 1) / 24,
        generated_encoded_duration_seconds=frames / 24,
        training_admission=dict(ppo=False, reason='No action/depth conditioning or geometric verification'))
    (out / 'request.json').write_text(json.dumps(evidence, indent=2))
    started = time.monotonic()
    try:
        print('LOADING', args.case, flush=True)
        pipe = load_pipeline(config['base_model'], config['weights'],
            memory_fraction=config['memory_fraction'])
        load_seconds = time.monotonic() - started
        inputs = dict(prompt=config['prompt'], image=Image.open(first).convert('RGB'),
            num_frames=frames, height=height, width=width, num_inference_steps=config['steps'],
            generator=torch.Generator().manual_seed(config['seed']), output_type='np')
        if last:
            inputs['last_image'] = Image.open(last).convert('RGB')
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        print('GENERATING', args.case, 'load_seconds', load_seconds, flush=True)
        with torch.inference_mode():
            result = pipe(**inputs, output=['videos'])
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        video = result['videos'][0]
        # Official modular pipeline returns RGB PIL images; accept float arrays
        # explicitly, checking shape/range instead of silently rescaling bytes.
        array = np.asarray([np.asarray(frame) for frame in video])
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError('Generated floating video outside finite [0,1]')
            array = np.rint(array * 255).astype(np.uint8)
        if array.shape != (frames, height, width, 3) or array.dtype != np.uint8:
            raise ValueError(f'Unexpected output {array.shape} {array.dtype}')
        # Save a lossless tensor for metrics; MP4 is a convenient inspection copy.
        np.save(out / 'frames.npy', array)
        with av.open(str(out / 'generated.mp4'), 'w') as container:
            stream = container.add_stream('libx264', rate=24)
            stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'fast'}
            for frame in array:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        metrics = compare_clip(array, references)
        metrics.update(status='complete', seed=config['seed'], frames=frames, steps=config['steps'],
            load_seconds=load_seconds, inference_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            frame_sha256=hashlib.sha256(array.tobytes()).hexdigest())
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print('PILOT_COMPLETE', json.dumps(metrics), flush=True)
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(exc),
            elapsed_seconds=time.monotonic() - started, case=args.case), indent=2))
        raise


if __name__ == '__main__':
    main()
