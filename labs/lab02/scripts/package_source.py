"""Rebuild upload source snapshots from current packages; never include outputs/data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def package(root=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    destination=root/'outputs/kaggle_upload'
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('lab02_standard_source.zip','lab02_optuna_source.zip'):
        with ZipFile(destination/name,'w',ZIP_DEFLATED) as archive:
            for directory in ('src','config','tests'):
                for path in sorted((root/directory).rglob('*')):
                    if path.is_file() and path.suffix in ('.py','.json') and '__pycache__' not in path.parts:
                        archive.write(path,path.relative_to(root).as_posix())
        print(destination/name)
    return destination


if __name__=='__main__':
    package()
