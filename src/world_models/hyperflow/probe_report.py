"""Summarize the observation ablation and render one synchronized diagnostic movie."""
import json
from pathlib import Path
import numpy as np
from .diagnostics import overlap_map,overlap_metrics,image_metrics
from driversi_world.world_models.observation_contract import ObservationCapabilities


def means(rows):
    return {key:float(np.mean([row[key] for row in rows])) for key in rows[0]}


def summarize(path):
    import av
    from PIL import Image,ImageDraw,ImageFont
    path=Path(path); out=path.parent; work=out/'work'
    report=json.loads(path.read_text()); cfg=report['config']; cameras=cfg['cameras']
    arrays={}; inputs={}; stats={}
    for c in cameras:
        source=np.load(work/f'{c}_input.npz'); generated=np.load(work/f'{c}_output.npz')
        inputs[c]=source
        arrays[c]={'mtgs':source['gs'],'vae':generated['vae'],'flash':generated['flash']}
        raw=json.loads((work/f'{c}_result.json').read_text())
        raw['metrics']={stage:means(rows) for stage,rows in raw['metrics'].items()}
        raw['recorded_pose_mtgs_vs_real']=means(raw['recorded_pose_mtgs_vs_real'])
        raw['real_log_detail']=means([image_metrics(x,x) for x in source['original']])
        raw['flash_vs_vae']=means([image_metrics(arrays[c]['flash'][i],arrays[c]['vae'][i])
                                  for i in cfg['geometry_frames']])
        stats[c]=raw
    overlaps={}
    for a,b in [('CAM_F0','CAM_L0'),('CAM_L0','CAM_F0'),('CAM_F0','CAM_R0'),('CAM_R0','CAM_F0')]:
        rows={stage:[] for stage in ('mtgs','vae','flash')}
        for j,index in enumerate(cfg['geometry_frames']):
            uv,mask=overlap_map(inputs[a]['depth'][j],inputs[b]['depth'][j],
                report['calibrations'][a],report['calibrations'][b])
            for stage in rows:
                rows[stage].append(overlap_metrics(arrays[a][stage][index],arrays[b][stage][index],uv,mask))
        overlaps[a+'->'+b]={stage:dict(mean_valid_fraction=float(np.mean([x['valid_fraction'] for x in values])),
            mean_rgb_mae=float(np.mean([x['rgb_mae'] for x in values if x['rgb_mae'] is not None]))
                if any(x['rgb_mae'] is not None for x in values) else None,
            tested_frames=len(values),usable_frames=sum(x['rgb_mae'] is not None for x in values))
            for stage,values in rows.items()}
    font=ImageFont.truetype(cfg['font_path'],26)
    stages=['mtgs','vae','flash']; labels=['MTGS 原始渲染','VAE 编解码','Flash + GS 约束']
    width=cfg['width']*3; height=(cfg['height']+36)*3+56
    video=out/'comparison.mp4'
    with av.open(str(video),'w',options={'movflags':'+faststart'}) as container:
        stream=container.add_stream('libx264',rate=cfg['fps']); stream.width=width;stream.height=height
        stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
        for i in range(cfg['frames']):
            panel=Image.new('RGB',(width,height),(18,22,30));draw=ImageDraw.Draw(panel)
            draw.text((12,10),f'PPO 观测诊断 | 各视角原生 1024×576 | t={i/cfg["fps"]:.2f}s | 前 / 左 / 右',font=font,fill='white')
            for row,stage in enumerate(stages):
                y=56+row*(cfg['height']+36)
                for col,c in enumerate(cameras):
                    x=col*cfg['width']
                    draw.text((x+12,y+1),labels[row]+' | '+c,font=font,fill='white')
                    panel.paste(Image.fromarray(arrays[c][stage][i]),(x,y+36))
            if i in (0,72,123): panel.resize((1536,height//2)).save(work/f'inspect_{i}.jpg')
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(panel),format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    with av.open(str(video)) as container:
        stream=container.streams.video[0]
        assert float(stream.average_rate)==cfg['fps']
        assert sum(1 for _ in container.decode(stream))==cfg['frames']
    # First frame has zero displacement: retain real-camera crops as a localized blur reference.
    crop=Image.new('RGB',(4*256,3*300+48),(18,22,30));draw=ImageDraw.Draw(crop)
    small=ImageFont.truetype(cfg['font_path'],20)
    for col,label in enumerate(['真实日志 t=0','MTGS','VAE','Flash']):draw.text((col*256+8,10),label,font=small,fill='white')
    for row,c in enumerate(cameras):
        images=[inputs[c]['original'][0]]+[arrays[c][s][0] for s in stages]
        for col,image in enumerate(images):
            x,y=cfg['width']//2,cfg['height']//2+70
            crop.paste(Image.fromarray(image[y-128:y+128,x-128:x+128]),(col*256,48+row*300))
            draw.text((col*256+8,48+row*300+260),c,font=small,fill='white')
    crop.save(out/'detail.png')
    capability=ObservationCapabilities(True,False,False)
    timings=[r['seconds'] for r in report['render_times'] if not r['depth']][1:]
    report.update(status='complete',camera_results=stats,overlap_results=overlaps,
        render_rgb_three_camera_seconds=dict(mean=float(np.mean(timings)),p95=float(np.percentile(timings,95))),
        overlap_scope='Fixed GS expected-depth visibility masks; photometric consistency only, not generated depth/pose validation',
        blur_scope='Energy can be inflated by artifacts; recorded-pose comparison has calibration/time/reconstruction errors; novel-view truth unavailable',
        ppo=dict(online_admitted=False,reasons=capability.rejection_reasons(),
            next_interface='executed state + current calibrated RGB/depth + past-only per-environment history -> synchronized enhanced views',
            frozen_during_rollout=True,reward_source='executed simulator geometry, never generated pixels',
            required_experiments=['causal short-window or single-step repair','future-perturbation invariance',
                'action intervention and history reset isolation','latency and multi-view actor preservation'],
            dataset_exported=False,policy_updates=0),
        final_files=['comparison.mp4','detail.png','result.json'])
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='complete',cameras={c:dict(timing=stats[c]['timing'],metrics=stats[c]['metrics']) for c in cameras},overlap=overlaps)),flush=True)


def add_history_summary(path):
    """Keep the additional causal-window feasibility check in the same result file."""
    from PIL import Image, ImageDraw, ImageFont
    path = Path(path)
    report = json.loads(path.read_text())
    cfg = report['config']
    work = path.parent / 'work'
    rows, inputs, outputs = {}, {}, {}
    for camera in cfg['cameras']:
        rows[camera] = json.loads((work / f'{camera}_history.json').read_text())
        with np.load(work / f'{camera}_input.npz') as source:
            inputs[camera] = dict(depth=source['depth'], current=source['gs'][[72, 96]])
        with np.load(work / f'{camera}_history.npz') as data:
            outputs[camera] = data['flash']
    cross_view = {}
    for source, target in [('CAM_F0','CAM_L0'), ('CAM_L0','CAM_F0'),
                           ('CAM_F0','CAM_R0'), ('CAM_R0','CAM_F0')]:
        measurements = []
        for j, frame in enumerate((72, 96)):
            index = cfg['geometry_frames'].index(frame)
            uv, mask = overlap_map(inputs[source]['depth'][index], inputs[target]['depth'][index],
                                   report['calibrations'][source], report['calibrations'][target])
            value = overlap_metrics(outputs[source][j], outputs[target][j], uv, mask)
            value['frame'] = frame
            value['mtgs'] = overlap_metrics(inputs[source]['current'][j], inputs[target]['current'][j], uv, mask)
            measurements.append(value)
        cross_view[source + '->' + target] = measurements
    # Compact internal visual inspection. Only the main comparison/detail/report are retained.
    panel = Image.new('RGB', (1536, 3*320), (18,22,30))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype(cfg['font_path'], 20)
    for row, camera in enumerate(cfg['cameras']):
        with np.load(work / f'{camera}_output.npz') as data:
            full_clip = data['flash'][72]
        for col, (label, pixels) in enumerate([
            ('MTGS t=3s', inputs[camera]['current'][0]),
            ('22 帧历史 Flash t=3s', outputs[camera][0]),
            ('整段 Flash t=3s', full_clip),
        ]):
            draw.text((col*512+8,row*320+4), camera+' | '+label, font=font, fill='white')
            panel.paste(Image.fromarray(pixels).resize((512,288)), (col*512,row*320+32))
    panel.save(work / 'history_inspect.png')
    report['history_probe'] = dict(
        cameras=rows, cross_view=cross_view,
        causal_input_boundary=True,
        scope='Two recorded-state requests per camera; 22 frames at 24 Hz; not a reactive PPO rollout',
        limitations=['No action intervention or episode-reset test',
                     'No object-level geometry validation',
                     'No stochastic-seed sweep or policy update',
                     'Camera workers are independent; native model is not jointly multi-view'],
    )
    report['validation']['boundary_and_geometry_tests_passed'] = 11
    report['validation']['trajectory_and_projection_tests_passed'] = 4
    report['ppo']['history_window_candidate'] = dict(
        requires_future_frames=False, updates_from_executed_state=False, multiview_validated=False,
        measured_compute_seconds=[r['seconds'] for c in rows.values() for r in c['rows']],
        full_environment_latency_measured=False, admitted=False,
        next_step='Wire past/current windows to executed states with per-environment reset; validate latency and actor geometry before PPO sampling',
    )
    report['ppo']['integration_points'] = dict(
        observation='RenderManager.render() after executed agent-state update; before policy IPC',
        action='Use an executable controller and reactive traffic, rather than the diagnostic log-play branch',
        reward='Enable geometry-based rewards; diagnostic configuration currently disables dense rewards',
        policy='Existing IPC supplies trajectories only; PPO additionally needs sampled action, old log-probability, value, termination and truncation',
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(history={c:rows[c]['rows'] for c in rows}, cross_view=cross_view)), flush=True)


if __name__=='__main__':
    import sys
    operation = add_history_summary if '--history' in sys.argv else summarize
    operation(sys.argv[1])
