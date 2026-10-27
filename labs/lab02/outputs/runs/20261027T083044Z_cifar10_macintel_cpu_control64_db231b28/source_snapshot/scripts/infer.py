"""Checkpoint inference CLI; invoked by README and delivery.complete_delivery.

Uses the same data path policy as training; importing this module does no work.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.delivery import resolve_data_dir


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image')
    parser.add_argument('--sample-index',type=int,default=0)
    parser.add_argument('--data-dir',type=Path)
    args=parser.parse_args(argv)
    import torch
    from PIL import Image
    from torchvision.datasets import CIFAR10
    from src.data import image_transform
    from src.models import build_model
    from src.eda import CLASSES
    torch.set_num_threads(4)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if saved['metadata'].get('dataset')!='CIFAR10':
        raise ValueError('This CLI requires a CIFAR10 checkpoint')
    name=saved['metadata']['model']
    model=build_model(name,pretrained=False)
    model.load_state_dict(saved['state_dict']);model.eval()
    if args.image:
        with Image.open(args.image) as source: image=source.convert('RGB')
        actual=None
    else:
        raw=CIFAR10(str(resolve_data_dir(ROOT,args.data_dir)),train=False)
        image,actual=raw[args.sample_index]
    with torch.inference_mode():
        x=image_transform(name,saved['metadata']['image_size'])(image).unsqueeze(0)
        probabilities=model(x).softmax(1)[0]
    prediction=int(probabilities.argmax())
    result=dict(model=name,predicted=CLASSES[prediction],probability=float(probabilities[prediction]),
        actual=CLASSES[actual] if actual is not None else None,
        probabilities=dict(zip(CLASSES,probabilities.tolist())))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    main()
