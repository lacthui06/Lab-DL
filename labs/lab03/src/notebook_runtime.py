"""Môi trường và hiển thị cho hai notebook; thuật toán nằm trong module chuyên biệt."""
from pathlib import Path
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone


class NotebookSession:
    def __init__(self, repo, mode='view', device='auto'):
        if mode not in {'view', 'prepare', 'train'} or device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('Invalid notebook mode or device')
        self.repo = Path(repo).resolve()
        self.lab = self.repo/'labs/lab03'
        self.mode, self.device = mode, device
        self.kaggle = Path('/kaggle/working').is_dir()
        self.python = sys.executable
        self.env = os.environ.copy()
        self.env.update(TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                        CUDA_VISIBLE_DEVICES='' if device == 'cpu' else '0',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8')
        self.env['LAB03_SESSION_ID'] = uuid.uuid4().hex
        self.paths = {}
        if mode == 'view':
            aliases = {'frozen_trial':'model', 'exercise1':'exercise1_current',
                       'exercise1_analysis':'exercise1_visuals'}
            for name in ('latest_preparation.json', 'latest_report.json'):
                selection = self.lab/'outputs'/name
                if not selection.is_file():continue
                for role,run_id in self.read(selection)['runs'].items():
                    if not isinstance(run_id,str) or run_id in {'.','..'} or Path(run_id).name != run_id:
                        raise ValueError('Unsafe run ID in selected artifacts')
                    self.paths[aliases.get(role,role)] = self.lab/'outputs/runs'/run_id
            if 'model' in self.paths:
                self.paths.setdefault('data',self.paths['model'])
                self.paths.setdefault('augmentation',self.paths['model'])
            if not self.paths:
                raise FileNotFoundError('Chưa có artifact được chọn; chạy MODE=prepare hoặc MODE=train trước.')

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def setup(self):
        """Kaggle dùng interpreter riêng; local dùng kernel đã cài requirements."""
        if self.mode == 'view':
            print('MODE=view: đọc artifacts đã lưu; không cài/tải model, không training.')
            return
        if self.kaggle:
            subprocess.run([sys.executable,'-m','pip','install','-q','uv==0.9.5'],check=True)
            env = Path('/kaggle/working/lab03-py312')
            self.python = str(env/'bin/python')
            def uv(*args):
                subprocess.run([sys.executable,'-m','uv',*map(str,args)],check=True)
            uv('python','install','3.12')
            if not Path(self.python).is_file():uv('venv','--python','3.12',env)
            uv('pip','install','--python',self.python,'--link-mode','copy','-r',self.lab/'requirements.txt')
            uv('pip','check','--python',self.python)
        self.run('-c','import sys,torch; print("Python:",sys.version); print("PyTorch:",torch.__version__); print("CUDA available:",torch.cuda.is_available())')
        print('Mode:',self.mode,'Device request:',self.device,'Artifacts:',self.lab/'outputs')

    def run(self, *args, env=None):
        subprocess.run([self.python,'-u',*map(str,args)],cwd=self.repo,
                       env=self.env if env is None else env,check=True)

    def active(self, kind):
        registry = self.read(self.lab/'config/workflow.json')
        run_id = registry['active_runs'][kind]
        if Path(run_id).name != run_id:raise ValueError('Unsafe run ID')
        return self.lab/'outputs/runs'/run_id

    def prepare(self):
        if self.mode == 'view':return
        config = self.lab/'config/frozen_trial.json'
        self.run('labs/lab03/scripts/preview_dataset.py','--config',config)
        self.paths['data'] = self.active('dataset_preview')
        self.run('labs/lab03/scripts/preview_augmentation.py','--config',config)
        self.paths['augmentation'] = self.active('augmentation_preview')
        from .utils.runs import save_json
        save_json(self.lab/'outputs/latest_preparation.json',{'runs':{
            role:self.paths[role].name for role in ('data','augmentation')}})

    def preflight(self):
        if self.mode != 'train':
            print('View mode: dùng bằng chứng đã lưu; không chạy lại tests hoặc training.')
            return
        folder = self.lab/'outputs/preflight'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        folder.mkdir(parents=True,exist_ok=False)
        env = self.env.copy();env['CUDA_VISIBLE_DEVICES']=''
        # Pytest tạo session fixtures riêng; không kế thừa ID của phiên training.
        env.pop('LAB03_SESSION_ID',None)
        with (folder/'pytest.log').open('w',encoding='utf-8') as log:
            process = subprocess.Popen([self.python,'-u','-m','pytest','labs/lab03/tests','-q',
                f'--junitxml={folder/"pytest.xml"}'],cwd=self.repo,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:print(line,end='');log.write(line)
            if process.wait():raise RuntimeError(f'Preflight failed: {folder}')
        print('Preflight PASSED:',folder)

    def exercise1(self):
        if self.mode == 'train':
            self.run('labs/lab03/scripts/run_exercise1.py','--config',self.lab/'config/frozen_trial.json')
            self.paths['exercise1_current'] = self.active('exercise1')
            self.paths['exercise1_visuals'] = self.active('exercise1_analysis')

    def train(self):
        if self.mode == 'train':
            self.run('labs/lab03/scripts/run_frozen_trial.py','--config',
                     self.lab/'config/frozen_trial.json','--device',self.device)
            self.paths['model'] = self.active('frozen_trial')
        else:print('View mode: đọc kết quả được chọn trong outputs/latest_report.json.')

    def table(self, role, filename, field=None, columns=None):
        import pandas as pd
        value = self.read(self.paths[role]/filename)
        if field is not None:value = value[field]
        frame = pd.DataFrame(value if isinstance(value,list) else [value])
        return frame[columns] if columns else frame

    def show(self, role, names):
        from IPython.display import display, Image, Markdown
        shown = []
        for name in names:
            path = self.paths[role]/name
            if not path.is_file():raise FileNotFoundError(path)
            display(Markdown(f'**{name}**'));display(Image(filename=str(path)));shown.append(path)
        return shown

    def performance(self):
        self.paths['performance'] = self.paths['model']
        return self.read(self.paths['performance']/'performance_summary.json')

    def export_preparation(self):
        """Xuất ảnh/log EDA đã xác minh để xem và làm report không cần tải lại data."""
        import zipfile
        import shutil
        from .evidence import verify_manifest
        from .utils.runs import save_json, hash_file
        if self.mode == 'view':
            print('Ảnh EDA đã lưu:', self.paths['data'])
            return
        roles = {role: self.paths[role] for role in ('data', 'augmentation')}
        for folder in roles.values():
            verify_manifest(folder)
        published = self.lab/'outputs/preparation'
        if published.exists(): shutil.rmtree(published)
        published.mkdir(parents=True, exist_ok=True)
        lines = ['# EDA và feature engineering', '', 'Ảnh và log từ các run nguồn đã kiểm tra SHA256.', '']
        files = [self.lab/'outputs/latest_preparation.json']
        for role, folder in roles.items():
            for path in sorted(folder.rglob('*')):
                if not path.is_file(): continue
                files.append(path)
                target = published/role/path.relative_to(folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                files.append(target)
                relative = target.relative_to(published).as_posix()
                if path.suffix == '.png': lines += [f'## {role}: {path.stem}', '', f'![{path.stem}]({relative})', '']
                elif path.name == 'console.log': lines += [f'[Log {role}]({relative})', '']
        (published/'index.md').write_text('\n'.join(lines), encoding='utf-8')
        save_json(published/'manifest.json', {p.relative_to(published).as_posix(): hash_file(p)
            for p in published.rglob('*') if p.is_file() and p.name != 'manifest.json'})
        files += [published/'index.md', published/'manifest.json']
        destination = Path('/kaggle/working') if self.kaggle else self.lab/'outputs'
        archive = destination/f'lab03_preparation_{roles["data"].name}.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            manifest = {}
            for path in files:
                name = path.relative_to(self.lab).as_posix()
                z.write(path, name); manifest[name] = hash_file(path)
            z.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
        self._download(archive)
        return archive

    def _download(self, path):
        from IPython.display import display, HTML, FileLink
        from urllib.parse import quote
        print(path, f'{path.stat().st_size/1024**2:.1f} MiB')
        if self.kaggle:
            display(HTML(f'<a href="/files/{quote(path.name)}" download="{path.name}">Tải {path.name}</a>'))
        else:
            display(FileLink(os.path.relpath(path, Path.cwd())))

    def export(self):
        if self.mode == 'view':
            print('Artifact report:',self.lab/'outputs/published/index.md')
            return
        from IPython.display import display, HTML, FileLink
        from urllib.parse import quote
        import hashlib, zipfile
        run = self.paths['model']
        destination = Path('/kaggle/working') if self.kaggle else self.lab/'outputs'
        review = destination/f'lab03_review_{run.name}.zip'
        checkpoint = destination/f'lab03_checkpoint_{run.name}.zip'
        with zipfile.ZipFile(review) as z:
            manifest = json.loads(z.read('bundle_manifest.json'))
            if not all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items()):
                raise ValueError('Review ZIP checksum mismatch')
        print('SHA256 review manifest: PASSED')
        for path in (review,checkpoint):
            print(path,f'{path.stat().st_size/1024**2:.1f} MiB')
            if self.kaggle:display(HTML(f'<a href="/files/{quote(path.name)}" download="{path.name}">Tải {path.name}</a>'))
            else:display(FileLink(os.path.relpath(path,Path.cwd())))
        print('Nếu link không tải: dùng trình duyệt file Output/Kaggle hoặc thư mục outputs/ trên local.')
