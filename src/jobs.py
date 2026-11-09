"""Independent synthesis stages and sealed file receipts for external orchestrators."""
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .data.selection import qualify, inspect_scenarios
from .runtime.files import digest, write_json
from .runtime.paths import configure_runtime, experiment_path


def verify_receipt(path, _seen=None):
    path = experiment_path(path)
    seen = set() if _seen is None else set(_seen)
    if str(path) in seen:
        raise ValueError("Cyclic World evidence dependency")
    seen.add(str(path))
    receipt = json.loads(path.read_text())
    root = experiment_path(receipt['root'])
    if receipt.get('schema') != 'driversi.world.receipt.v1' or path.parent != root:
        raise ValueError('Foreign World receipt')
    if receipt.get('status') != 'completed':
        raise ValueError('World stage did not complete')
    for parent in receipt.get("dependencies", []):
        if digest(parent["path"]) != parent["sha256"]:
            raise ValueError("World dependency receipt changed")
        prior = verify_receipt(parent["path"], seen)
        if prior.get("selection_sha256") != receipt.get("selection_sha256"):
            raise ValueError("World stages use different source selections")
    for relative, item in receipt['files'].items():
        file = (root/relative).resolve(strict=True)
        if not file.is_relative_to(root) or file.stat().st_size != item['size'] or digest(file) != item['sha256']:
            raise ValueError('World evidence changed: '+relative)
    for file in receipt['artifacts'].values():
        key = Path(file).relative_to(root).as_posix()
        if key not in receipt['files']:
            raise ValueError('Unsealed World artifact')
    return receipt


def _seal(output, stage, status, summary, artifacts, elapsed, error=None, spec=None):
    files={}
    for path in output.rglob('*'):
        if not path.is_file() or path.name=='receipt.json':
            continue
        relative=path.relative_to(output)
        if relative.parts[0] in ('cache','tmp'):
            continue
        if not path.resolve().is_relative_to(output):
            raise ValueError('World output contains an external symlink')
        files[relative.as_posix()]={'size':path.stat().st_size,'sha256':digest(path)}
    dependencies = ([{'path':spec['input_receipt'], 'sha256':digest(spec['input_receipt'])}]
                    if spec and spec.get('input_receipt') else [])
    receipt=dict(selection_sha256=(spec or {}).get('selection_sha256'), dependencies=dependencies, schema='driversi.world.receipt.v1',root=str(output),stage=stage,status=status,
                 summary=summary,artifacts=artifacts,files=files,wall_seconds=elapsed,
                 trainable=False,parameter_updates=0,error=error)
    write_json(output/'receipt.json',receipt)
    return receipt


def run(stage, spec, output):
    output=experiment_path(output)
    output.mkdir(parents=True,exist_ok=False)
    configure_runtime(output)
    started=time.monotonic()
    artifacts={}; summary={}
    try:
        if digest(spec['selection']) != spec['selection_sha256']:
            raise ValueError('Selection changed after preparation')
        selection=json.loads(Path(spec['selection']).read_text())
        qualify(selection,spec.get('protected_groups',()))
        if stage=='convert':
            filtered=output/'scene_filter.yaml'
            filtered.write_text(yaml.safe_dump({'tokens':[r['token'] for r in selection['records']],
                 'log_names':sorted({r['log_name'] for r in selection['records']})}))
            command=[sys.executable,'-B','-m',
                'driversi_world.simulation.utils.dataset_utils.nuplan.digitaltwin_nuplan_converter_navsim_filter',
                '--navsim-filters',str(filtered),'--digitaltwin-asset-root',spec['asset_root'],
                '--nuplan-root-path',spec['nuplan_root'],'--nuplan-db-path',spec['nuplan_db'],
                '--nuplan-map-root',spec['nuplan_maps'],'--openscene-dataroot',spec['openscene_root'],
                '--sample-interval','10','--num-processes',str(spec['cpu_workers']),
                '--out-dir',str(output/'scenarios')]
            subprocess.run(command,cwd=output,check=True,timeout=spec['max_wall_seconds'])
            scenes=output/'scenarios/all_scenarios.pkl'
            summary=inspect_scenarios(scenes,selection)
            if summary['missing'] or summary['rejected']:
                raise ValueError('Incomplete or unsupported scene window')
            artifacts={'scenarios':str(scenes)}
        elif stage=='rollout':
            previous=verify_receipt(spec['input_receipt'])
            if previous.get('selection_sha256') != spec['selection_sha256']:
                raise ValueError('World source selection differs across stages')
            if previous['stage']!='convert':
                raise ValueError('Rollout requires converted scenarios')
            source=previous['artifacts']['scenarios']
            summary=inspect_scenarios(source,selection)
            if summary['missing'] or summary['rejected']:
                raise ValueError('Unqualified scene window')
            expected=[r['token'] for r in selection['records']]
            summary={'completed':[],'scored':[],'expected':expected,'traffic_mode':spec['traffic_mode']}
            from .simulation.execution import run as simulate
            native=output/'native';native.mkdir()
            simulate({**spec,'scenarios':source,'output':str(native)})
            report=native/'runner_report.json'
            rows=json.loads(report.read_text())
            completed=[r['scenario_name'].rsplit('-',1)[-1] for r in rows if r['succeeded']]
            summary['completed']=completed
            summary['native_rows']=rows
            if any(not r['succeeded'] for r in rows) or set(completed)!=set(expected):
                raise ValueError('Incomplete native rollout')
            from .simulation.export import export_candidates
            card=export_candidates(native,selection,output)
            summary['scored']=[r['source']['token'] for r in card['records']]
            artifacts={'rollout':str(report),'candidates':str(output/'candidates.json')}
        elif stage=='export':
            previous=verify_receipt(spec['input_receipt'])
            if previous.get('selection_sha256') != spec['selection_sha256']:
                raise ValueError('World source selection differs across stages')
            if previous['stage']!='rollout':
                raise ValueError('Export requires a complete scored rollout')
            from .simulation.export import export_candidates
            card=export_candidates(Path(previous['artifacts']['rollout']).parent,selection,output)
            summary=card
            artifacts={'candidates':str(output/'candidates.json')}
        else:
            raise ValueError('Unknown World stage')
        return _seal(output,stage,'completed',summary,artifacts,time.monotonic()-started,spec=spec)
    except Exception as exc:
        return _seal(output,stage,'failed',summary,{},time.monotonic()-started,
                     type(exc).__name__+': '+str(exc),spec=spec)
