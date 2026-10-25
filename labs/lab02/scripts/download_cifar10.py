"""Download official CIFAR-10 with bounded parallel byte ranges and MD5 verification."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import urllib.request
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
SIZE = 170498071
DEST = ROOT / 'data' / 'cifar-10-python.tar.gz'
PARTS = ROOT / 'data' / 'cifar_download_parts'


def part(bounds):
    start, end = bounds
    dest = PARTS / str(start)
    if dest.exists() and dest.stat().st_size == end - start + 1:
        return dest
    for attempt in range(4):
        try:
            request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise RuntimeError('Server did not honor requested byte range')
                content = response.read()
            if len(content) != end - start + 1:
                raise RuntimeError('Incomplete part')
            dest.write_bytes(content)
            print(f'Part {start // (4*1024*1024) + 1}/41 ready', flush=True)
            return dest
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


if __name__ == '__main__':
    PARTS.mkdir(parents=True, exist_ok=True)
    chunks = [(s, min(s + 4*1024*1024 - 1, SIZE-1)) for s in range(0, SIZE, 4*1024*1024)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(part, chunks))
    temporary = DEST.with_suffix('.verified.tmp')
    digest = hashlib.md5()
    with temporary.open('wb') as output:
        for path in paths:
            data = path.read_bytes()
            digest.update(data)
            output.write(data)
    if digest.hexdigest() != 'c58f30108f718f92721af3b95e74349a':
        raise RuntimeError('Official CIFAR10 archive MD5 mismatch')
    temporary.replace(DEST)
    from torchvision.datasets import CIFAR10
    for train in (True, False):
        CIFAR10(str(ROOT / 'data'), train=train, download=True)
    print('Official CIFAR10 verified and extracted.', flush=True)
