"""Re-evaluate locked Lab02 checkpoints without changing the original run."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2] / 'labs/lab02'
sys.path.insert(0, str(ROOT))
from src.datasets.classification import image_loader
from src.utils.artifacts import resolve_data_dir
from src.training.trainer import predict, seed_all
from src.models.backbones import build_model
from src.utils.plots import export_evaluation


def main():
    run = Path(json.loads((ROOT / 'outputs/selected_run.json').read_text())['run'])
    cfg = json.loads((run / 'config.json').read_text())
    selected = np.load(run / 'selected_indices.npz')['test']
    output = Path(__file__).resolve().parent / run.name
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['threads'])
    seed_all(cfg['seed'])
    device = torch.device('cpu')
    results = []
    for name in cfg['models']:
        for stage in ['head_selected', 'finetune']:
            source = run / name / stage
            checkpoint = source / 'best.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            dropout = saved['metadata'].get('dropout', 0.)
            model = build_model(name, pretrained=False, dropout=dropout)
            model.load_state_dict(saved['state_dict'])
            model.eval()
            loader = image_loader(resolve_data_dir(ROOT), 'test', selected, name,
                                  64, cfg['seed'], image_size=cfg['image_size'])
            print(f'Evaluating {name}/{stage}: {len(selected)} samples', flush=True)
            probabilities, labels, indices = predict(model, loader, device)
            old = np.load(source / 'test_predictions.npz')
            np.testing.assert_array_equal(indices, old['indices'])
            np.testing.assert_array_equal(labels, old['labels'])
            delta = float(np.abs(probabilities - old['probabilities']).max())
            destination = output / name / stage
            destination.mkdir(parents=True, exist_ok=True)
            metrics = export_evaluation(destination, probabilities, labels, indices,
                                        resolve_data_dir(ROOT))
            sample = next(iter(loader))[0][:8]
            with torch.inference_mode():
                before = model(sample).softmax(1)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'roundtrip.pt'
                torch.save({'state_dict': model.state_dict(), 'metadata': saved['metadata']}, path)
                loaded = torch.load(path, map_location='cpu', weights_only=True)
                restored = build_model(name, pretrained=False, dropout=dropout)
                restored.load_state_dict(loaded['state_dict'])
                restored.eval()
                with torch.inference_mode():
                    error = float((before - restored(sample).softmax(1)).abs().max())
            row = dict(model=name, stage=stage, metrics=metrics,
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       previous_probability_max_error=delta,
                       same_argmax_predictions=bool(np.array_equal(probabilities.argmax(1), old['probabilities'].argmax(1))),
                       roundtrip_samples=len(sample), roundtrip_max_error=error,
                       roundtrip_passed=error <= 1e-6)
            results.append(row)
            (output / 'verification.json').write_text(json.dumps(dict(run=str(run), results=results), indent=2))
            print(json.dumps(row), flush=True)
            assert row['roundtrip_passed'], 'Save/load changed predictions'
    print('Evaluation completed:', output, flush=True)


if __name__ == '__main__':
    main()
