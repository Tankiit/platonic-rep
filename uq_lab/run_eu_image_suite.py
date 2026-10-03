"""Run class-withholding sensitivity for one exact Gaussian head family."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path('uq_runs'))
    p.add_argument('--head',choices=('linear','rbf'),required=True)
    p.add_argument('--classes',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--layers',type=int,nargs='+',default=[2])
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--max-train',type=int,default=1024)
    args=p.parse_args()
    script=Path(__file__).with_name('uq_eu_specific.py')
    for heldout in args.classes:
        out=args.output_root/f'eu_specific_image_{args.head}_class{heldout}'
        print(f'Starting {args.head} heldout={heldout}',flush=True)
        subprocess.run([sys.executable,str(script),'--mode','features','--head',args.head,
                        '--root',str(args.root),'--labels',str(args.root/'cifar10_train_labels.npy'),
                        '--heldout',str(heldout),'--layers',*[str(x) for x in args.layers],
                        '--seeds',*[str(x) for x in args.seeds],'--max-train',str(args.max_train),
                        '--output',str(out)],check=True)
    print(f'{args.head}: all held-out classes complete',flush=True)


if __name__=='__main__':
    main()
