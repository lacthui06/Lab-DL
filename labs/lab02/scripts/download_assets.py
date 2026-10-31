"""Asset CLI used by README: dataset/weights, with optional ranged download.

--help has no download side effects. --chunked replaces the old fallback CLI.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import sys
import time
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.artifacts import resolve_data_dir

ROOT=Path(__file__).resolve().parents[1]
URL='https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
SIZE=170498071
MD5='c58f30108f718f92721af3b95e74349a'


def download_archive(data_dir):
    """Bounded byte ranges, size/header verification and official MD5 before use."""
    data_dir=Path(data_dir)
    data_dir.mkdir(parents=True,exist_ok=True)
    destination=data_dir/'cifar-10-python.tar.gz'
    if destination.exists():
        with destination.open('rb') as stream:
            if hashlib.file_digest(stream,'md5').hexdigest()==MD5:
                return destination
    parts=data_dir/'cifar_download_parts'
    parts.mkdir(exist_ok=True)

    def part(bounds):
        start,end=bounds
        path=parts/str(start)
        if path.exists() and path.stat().st_size==end-start+1:
            return path
        for attempt in range(4):
            try:
                request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=90) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                        raise RuntimeError('Server did not honor requested byte range')
                    content=response.read()
                if len(content)!=end-start+1:
                    raise RuntimeError('Incomplete download part')
                path.write_bytes(content)
                return path
            except Exception:
                if attempt==3: raise
                time.sleep(2)

    bounds=[(s,min(s+4*1024*1024-1,SIZE-1)) for s in range(0,SIZE,4*1024*1024)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        paths=list(executor.map(part,bounds))
    temporary=destination.with_suffix('.verified.tmp')
    digest=hashlib.md5()
    with temporary.open('wb') as stream:
        for path in paths:
            content=path.read_bytes();digest.update(content);stream.write(content)
    if digest.hexdigest()!=MD5:
        raise RuntimeError('Official CIFAR-10 archive checksum mismatch')
    temporary.replace(destination)
    return destination


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,help='Explicit raw-data folder; otherwise LAB02_DATA or root/data')
    parser.add_argument('--chunked',action='store_true',help='Use checksum-verified ranged download for CIFAR-10')
    parser.add_argument('--dataset-only',action='store_true',help='Skip ImageNet weights')
    args=parser.parse_args(argv)
    data_dir=resolve_data_dir(ROOT,args.data_dir)
    if args.chunked: download_archive(data_dir)
    from torchvision.datasets import CIFAR10
    for train in (True,False): CIFAR10(str(data_dir),train=train,download=True)
    if not args.dataset_only:
        import torch
        from src.models.backbones import SPECS
        torch.hub.set_dir(str(ROOT/'data/pretrained'))
        for name,(_,weights) in SPECS.items():
            print('Downloading',name,flush=True)
            weights.get_state_dict(progress=True,check_hash=True)
    print('Official assets ready.')


if __name__=='__main__':
    main()
