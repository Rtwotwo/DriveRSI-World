"""One-camera 1024x576 MTGS / identical-latent VAE / Flash observation ablation."""
import json
import sys
import time
from pathlib import Path
from driversi_world.runtime.paths import configure_runtime,experiment_path


def run(path,camera):
    record=json.loads(Path(path).read_text()); cfg=record['config']
    out=experiment_path(cfg['output']); work=out/'work'; configure_runtime(work/camera)
    import numpy as np
    import torch
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from .blocks import flash_blocks
    from .loader import load_pipeline
    from .trajectory import install_projection
    from .diagnostics import image_metrics
    torch.set_num_threads(4)
    blocks=flash_blocks('fl2va'); projection=install_projection(blocks,cfg['projection_strength'])
    tick=time.perf_counter(); print('LOADING',camera,flush=True)
    pipe=load_pipeline(cfg['base_model'],cfg['weights'],blocks=blocks)
    load_seconds=time.perf_counter()-tick
    # Rendering may run concurrently with model loading, but no partial input is consumed.
    deadline=time.monotonic()+900
    while not (work/f'{camera}_input.npz').exists():
        if time.monotonic()>deadline: raise TimeoutError('Waiting for completed GS preparation')
        time.sleep(2)
    data=np.load(work/f'{camera}_input.npz'); gs=data['gs']
    timing={}; peaks={}
    def begin():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); return time.perf_counter()
    def end(name,start):
        torch.cuda.synchronize(); timing[name]=time.perf_counter()-start
        peaks[name]=torch.cuda.max_memory_allocated()
        print(name,camera,timing[name],flush=True)
    with torch.inference_mode():
        tick=begin()
        pixels=torch.from_numpy(gs.copy()).permute(3,0,1,2).unsqueeze(0).cuda()
        reference=encode_vae_condition(pipe.vae,pixels,pipe.pixel_mean,pipe.pixel_std,pipe.keyframe_encode_seed)
        del pixels
        end('reference_encode_seconds',tick)
        tick=begin()
        mean=torch.tensor(pipe.vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std=torch.tensor(pipe.vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        with torch.autocast('cuda',dtype=torch.float16):
            decoded=pipe.vae.decode(reference.cuda()*std+mean,return_dict=False)[0]
        pm=torch.tensor(pipe.pixel_mean,device='cuda').view(1,-1,1,1,1)
        ps=torch.tensor(pipe.pixel_std,device='cuda').view(1,-1,1,1,1)
        vae=np.rint((decoded.float()*ps+pm).clamp(0,1)[0].permute(1,2,3,0).cpu().numpy()*255).astype(np.uint8)
        del decoded
        end('vae_decode_seconds',tick)
        noise=torch.randn(reference.shape,generator=torch.Generator().manual_seed(cfg['seed']))
        projection.configure(reference,noise,pipe.patch_size)
        tick=begin()
        result=pipe(prompt=cfg['prompt'],image=Image.fromarray(gs[0]),last_image=Image.fromarray(gs[-1]),
            num_frames=cfg['frames'],height=cfg['height'],width=cfg['width'],num_inference_steps=cfg['steps'],
            latents=noise,generator=torch.Generator().manual_seed(cfg['seed']),output_type='np',output=['videos'])
        flash=np.asarray(result['videos'][0])
        if not np.isfinite(flash).all(): raise RuntimeError('Nonfinite generated pixels')
        flash=np.rint(flash*255).astype(np.uint8)
        end('flash_seconds',tick)
    assert vae.shape==flash.shape==gs.shape and len(projection.trace)==cfg['steps']
    # Lossless arrays are temporary measurement inputs, not exported training samples.
    np.savez_compressed(work/f'{camera}_output.npz',vae=vae,flash=flash)
    sample=cfg['geometry_frames']
    metrics={name:[image_metrics(array[i],gs[i]) for i in sample]
             for name,array in [('mtgs',gs),('vae',vae),('flash',flash)]}
    replay=[image_metrics(a,b) for a,b in zip(data['replay'],data['original'])]
    report=dict(camera=camera,load_seconds=load_seconds,timing=timing,peaks=peaks,metrics=metrics,
        recorded_pose_mtgs_vs_real=replay,projection_calls=len(projection.trace),seed=cfg['seed'],
        causal=False,uses_future_gs=True,uses_last_frame=True,independent_camera_generation=True)
    (work/f'{camera}_result.json').write_text(json.dumps(report,indent=2))
    print('CAMERA_COMPLETE',camera,flush=True)


def run_history_probe(path, camera):
    """Measure two current-frame requests using 22 *past/current* rendered frames.

    This exercises a causal input boundary on recorded renders, not a PPO rollout.
    The FL2VA model still generates a window; only its final observation is used.
    No model state or generated past frame is fed back into the next request.
    """
    from driversi_world.world_models.observation_contract import (
        past_window_indices, validate_observation_timestamps,
    )
    record = json.loads(Path(path).read_text())
    cfg = record['config']
    work = experiment_path(cfg['output']) / 'work'
    configure_runtime(work / (camera + '_history'))
    import numpy as np
    import torch
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from .blocks import flash_blocks
    from .loader import load_pipeline
    from .trajectory import install_projection
    from .diagnostics import image_metrics

    torch.set_num_threads(4)
    blocks = flash_blocks('fl2va')
    projection = install_projection(blocks, cfg['projection_strength'])
    tick = time.perf_counter()
    pipe = load_pipeline(cfg['base_model'], cfg['weights'], blocks=blocks,
                         diagnostic_short_window=True)
    load_seconds = time.perf_counter() - tick
    with np.load(work / f'{camera}_input.npz') as source:
        gs = source['gs']
    timestamps = (np.arange(len(gs)) / cfg['fps']).tolist()
    rows, observations = [], []
    for frame in (72, 96):
        current_time = timestamps[frame]
        indices = past_window_indices(timestamps, current_time, 22)
        validate_observation_timestamps(current_time, {camera: current_time}, timestamps[indices[-1]])
        window = gs[list(indices)].copy()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            pixels = torch.from_numpy(window).permute(3, 0, 1, 2).unsqueeze(0).cuda()
            reference = encode_vae_condition(pipe.vae, pixels, pipe.pixel_mean,
                                             pipe.pixel_std, pipe.keyframe_encode_seed)
            del pixels
            noise = torch.randn(reference.shape, generator=torch.Generator().manual_seed(cfg['seed']))
            projection.configure(reference, noise, pipe.patch_size)
            result = pipe(prompt=cfg['prompt'], image=Image.fromarray(window[0]),
                          last_image=Image.fromarray(window[-1]), num_frames=22,
                          height=cfg['height'], width=cfg['width'],
                          num_inference_steps=cfg['steps'], latents=noise,
                          generator=torch.Generator().manual_seed(cfg['seed']),
                          output_type='np', output=['videos'])
            output = np.asarray(result['videos'][0])
            if output.shape != window.shape or not np.isfinite(output).all():
                raise RuntimeError(f'Unexpected short-window output: {output.shape}')
            current = np.rint(output[-1].clip(0, 1) * 255).astype(np.uint8)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        observations.append(current)
        rows.append(dict(frame=frame, current_time=current_time, input_indices=indices,
                         seconds=elapsed, peak_bytes=torch.cuda.max_memory_allocated(),
                         projection_calls=len(projection.trace), metrics=image_metrics(current, gs[frame])))
        print('HISTORY_CURRENT_FRAME', camera, frame, elapsed, flush=True)
    np.savez_compressed(work / f'{camera}_history.npz', flash=np.stack(observations))
    result = dict(camera=camera, load_seconds=load_seconds, rows=rows,
                  conditioning='22 recorded GS frames ending at the current timestamp',
                  history_spacing_seconds=1/cfg['fps'], future_conditioning=False,
                  native_min_duration_seconds=5, duration_guard_relaxed_for_diagnostic=True,
                  native_five_frame_request='rejected by official duration guard on all three cameras',
                  relaxed_five_frame_request='native VAE decoder has zero complete chunks; failed on all three cameras',
                  measured_scope='encode + conditioned six-step generation + decode; model load excluded',
                  closed_loop_executed=False, dataset_exported=False)
    (work / f'{camera}_history.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    operation = run_history_probe if len(sys.argv) > 3 and sys.argv[3] == 'history' else run
    operation(sys.argv[1], sys.argv[2])
