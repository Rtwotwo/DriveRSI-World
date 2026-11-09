"""Independent World capability entry point (no policy/model imports)."""
import argparse
import json
from pathlib import Path
from .runtime.paths import configure_runtime
from .runtime.files import write_json

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect','mine','convert','rollout','export'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--scores',type=Path)
    parser.add_argument('--sources',type=Path)
    parser.add_argument('--threshold',type=float,default=.6)
    parser.add_argument('--limit',type=int,default=2)
    args=parser.parse_args(argv)
    if args.command in ('convert','rollout','export'):
        if args.config is None:
            parser.error('--config is required')
        from .jobs import run
        result=run(args.command,json.loads(args.config.read_text()),args.output)
        print(json.dumps({k:result[k] for k in ('status','root','stage','wall_seconds','trainable','error')}))
        return 0 if result['status']=='completed' else 2
    configure_runtime(args.output)
    if args.command=='mine':
        if args.scores is None or args.sources is None:
            parser.error('--scores and --sources are required')
        from .data.selection import mine
        print(json.dumps(mine(args.scores,args.sources,args.output,args.threshold,args.limit)))
        return 0
    result={'schema':'driversi.world.capabilities.v1','package':'driversi_world',
            'repository':str(Path(__file__).resolve().parents[1]),
            'simulations':0,'training_updates':0,'trainable':False}
    write_json(args.output/'capabilities.json',result)
    print(json.dumps(result))
    return 0
