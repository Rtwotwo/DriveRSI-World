"""Generate a GS-guided Flash clip and a compact original-log comparison video.

Outputs: comparison.mp4, original.mp4, flash.mp4 and one combined result.json.
The reference and original camera are never substituted for Flash's output.
"""
import hashlib
import json
import time
from pathlib import Path

from driversi_world.runtime.paths import configure_runtime, experiment_path


def write_video(path, frames, fps):
    """Write one H.264 artifact and verify every decoded frame and the frame rate."""
    import av
    with av.open(str(path), 'w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == fps
        count = 0
        for frame in container.decode(stream):
            assert (frame.height, frame.width) == frames.shape[1:3]
            count += 1
        assert count == len(frames)
    return dict(file=path.name, bytes=path.stat().st_size, frames=count,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def comparison_frames(original, generated, record, font_path):
    """Two equally sized views plus a metric trajectory schematic and source labels."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(font_path), 24)
    small = ImageFont.truetype(str(font_path), 19)
    title = ImageFont.truetype(str(font_path), 28)
    path = record['trajectory']
    base, changed = np.array(path['original_xy']), np.array(path['edited_xy'])
    heading = path['original_yaw'][0]
    origin = base[0]
    # One common metric scale for the two paths, looking forward from the initial ego.
    left = np.array([-np.sin(heading), np.cos(heading)])
    forward = np.array([np.cos(heading), np.sin(heading)])
    def project(xy):
        return np.column_stack(((xy-origin)@left, (xy-origin)@forward))
    a, b = project(base), project(changed)
    all_xy = np.concatenate([a, b])
    center = (all_xy.min(0) + all_xy.max(0)) / 2
    extent = np.maximum(np.ptp(all_xy, axis=0), 1)
    scale = min(190 / extent[0], 135 / extent[1])
    def screen(xy):
        return [(int(1120 - (p[0]-center[0])*scale), int(515 - (p[1]-center[1])*scale)) for p in xy]
    a, b = screen(a), screen(b)
    frames = []
    for i, (real, flash) in enumerate(zip(original, generated)):
        image = Image.new('RGB', (1280, 616), (17, 22, 31))
        draw = ImageDraw.Draw(image)
        draw.text((24, 14), '原始日志', font=title, fill='white')
        draw.text((664, 14), '新轨迹 · Flash 生成', font=title, fill=(84, 221, 222))
        image.paste(Image.fromarray(real).resize((640, 360), Image.Resampling.LANCZOS), (0, 64))
        image.paste(Image.fromarray(flash).resize((640, 360), Image.Resampling.LANCZOS), (640, 64))
        draw.line((640, 64, 640, 424), fill=(84, 221, 222), width=2)
        draw.text((24, 443), '真实相机帧：2 Hz，按时间保持显示', font=small, fill=(205, 213, 227))
        draw.text((24, 480), '生成条件：沿新轨迹渲染的整段 GS 序列', font=small, fill=(205, 213, 227))
        shift = path['lateral_offset_m'][i]
        seconds = path['times_seconds'][i]
        draw.text((24, 526), f't = {seconds:.2f} s    左移 {shift:.2f} m / 3.00 m', font=font, fill=(84, 221, 222))
        draw.text((24, 570), '视觉反事实演示 · 未验证几何一致性与交通安全', font=small, fill=(158, 170, 187))
        draw.text((875, 440), '轨迹俯视图', font=small, fill='white')
        draw.text((875, 476), '原轨迹', font=small, fill=(210, 216, 225))
        draw.text((875, 512), '新轨迹', font=small, fill=(84, 221, 222))
        draw.line(a, fill=(210, 216, 225), width=3)
        draw.line(b, fill=(84, 221, 222), width=3)
        for point, color in [(a[i], 'white'), (b[i], (84, 221, 222))]:
            x, y = point
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color)
        frames.append(np.asarray(image))
    return np.stack(frames)


def run(record_path):
    record_path = experiment_path(record_path)
    record = json.loads(record_path.read_text())
    config = record['config']
    out = experiment_path(config['output'])
    work = out / 'work'
    configure_runtime(work)
    import numpy as np
    import torch
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from .blocks import flash_blocks
    from .loader import load_pipeline
    from .trajectory import install_projection

    torch.set_num_threads(4)
    blocks = flash_blocks('fl2va')
    projection = install_projection(blocks, config['projection_strength'])
    start = time.monotonic()
    print('LOADING_FLASH', flush=True)
    pipe = load_pipeline(config['base_model'], config['weights'], blocks=blocks)
    loaded = time.monotonic()
    # GS rendering may run on another GPU while these large weights load.
    record = json.loads(record_path.read_text())
    if 'trajectory' not in record:
        raise RuntimeError('GS preparation has not finished; preserve the current attempt log')
    if record['trajectory'].get('heading_mode') != 'preserve_log_heading':
        raise RuntimeError('The corrected, heading-preserving camera sequence is required')
    data = np.load(work / 'inputs.npz', allow_pickle=False)
    original, gs = data['original'], data['gs']
    expected = (config['frames'], config['height'], config['width'], 3)
    if original.shape != expected or gs.shape != expected or gs.dtype != np.uint8:
        raise ValueError('Invalid original or GS reference sequence')
    print('ENCODING_GS_SEQUENCE', 'load_seconds', loaded-start, flush=True)
    with torch.inference_mode():
        pixels = torch.from_numpy(gs.copy()).permute(3, 0, 1, 2).unsqueeze(0).cuda()
        reference = encode_vae_condition(pipe.vae, pixels, pipe.pixel_mean, pipe.pixel_std,
                                         pipe.keyframe_encode_seed)
        del pixels
        noise = torch.randn(reference.shape, generator=torch.Generator().manual_seed(config['seed']))
        projection.configure(reference, noise, pipe.patch_size)
        torch.cuda.reset_peak_memory_stats()
        generated_start = time.monotonic()
        print('GENERATING_GS_GUIDED_FLASH', tuple(reference.shape), flush=True)
        result = pipe(prompt=config['prompt'], image=Image.fromarray(gs[0]),
            last_image=Image.fromarray(gs[-1]), num_frames=config['frames'],
            height=config['height'], width=config['width'], num_inference_steps=config['steps'],
            latents=noise, generator=torch.Generator().manual_seed(config['seed']),
            output_type='np', output=['videos'])
        torch.cuda.synchronize()
    generated = np.asarray(result['videos'][0])
    if generated.shape != expected or not np.isfinite(generated).all() or generated.min() < 0 or generated.max() > 1:
        raise ValueError(f'Unexpected generated video: {generated.shape}')
    generated = np.rint(generated * 255).astype(np.uint8)
    if np.ptp(generated) == 0 or len(projection.trace) != config['steps']:
        raise RuntimeError('Empty video or missing trajectory-projection calls')
    inference_seconds = time.monotonic() - generated_start
    # Save actual generated pixels, without compositing them with the GS reference.
    artifacts = [write_video(out / 'original.mp4', original, config['fps']),
                 write_video(out / 'flash.mp4', generated, config['fps'])]
    font_path = Path(config.get('font_path', work / 'font.ttc'))
    composite = comparison_frames(original, generated, record, font_path)
    artifacts.append(write_video(out / 'comparison.mp4', composite, config['fps']))
    # A single temporary contact sheet supports visual inspection before cleanup.
    indices = [0, len(composite)//3, 2*len(composite)//3, len(composite)-1]
    Image.fromarray(np.concatenate([composite[i] for i in indices], axis=0)).save(work/'inspection.jpg')
    record.update(status='complete', method='Flash with dense GS latent projection after every sampling step',
        method_scope='Experimental sampling-time guidance; not a trained pose/action controller',
        load_seconds=loaded-start, reference_encode_seconds=generated_start-loaded,
        inference_seconds=inference_seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        projection_steps=projection.trace, outputs=artifacts,
        rgb_mae_to_gs=float(np.abs(generated.astype(np.float32)-gs).mean()/255),
        generated_temporal_rgb_delta=float(np.abs(np.diff(generated.astype(np.float32), axis=0)).mean()/255),
        pixel_compositing_with_gs=False, geometry_verified=False, collision_verified=False,
        training_admitted=False, parameter_updates=0)
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('DEMO_COMPLETE', json.dumps(dict(inference_seconds=inference_seconds,
        rgb_mae_to_gs=record['rgb_mae_to_gs'], artifacts=artifacts)), flush=True)


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[1]))
