"""Checkpoint inference CLI; invoked by README and delivery.complete_delivery.

Uses the same data path policy as training; importing this module does no work.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.delivery import resolve_data_dir


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image')
    parser.add_argument('--sample-index',type=int,default=0)
    parser.add_argument('--data-dir',type=Path)
    args=parser.parse_args(argv)
    from src.inference import load_model, predict_image
    from torchvision.datasets import CIFAR10
    session=load_model(Path(args.checkpoint))
    if args.image:
        image=Path(args.image)
        actual=None
    else:
        raw=CIFAR10(str(resolve_data_dir(ROOT,args.data_dir)),train=False)
        image,actual=raw[args.sample_index]
    result=predict_image(session,image,actual)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    main()
