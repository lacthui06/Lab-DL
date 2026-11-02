"""Gắn policy/config vào checkpoint đã giải nén, xác minh weights trước khi ghi."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.evidence import read_json, verify_manifest
from labs.lab03.src.decision import checkpoint_identity, load_decision_policy
from labs.lab03.src.utils.runs import ROOT, save_json


def prepare_checkpoint(model_path, source_run):
    model_path = Path(model_path).resolve()
    source_run = Path(source_run).resolve()
    if model_path == source_run or source_run in model_path.parents:
        raise ValueError('Prepare an extracted checkpoint copy; never modify original run evidence')
    verify_manifest(source_run)
    expected = read_json(source_run/'frozen_selection.json')['weights_sha256']
    identity = checkpoint_identity(model_path)
    if identity['weights_sha256'] != expected:
        raise ValueError('Checkpoint weights SHA256 does not match the frozen trial')
    policy = read_json(source_run/'decision_threshold.json')
    policy['checkpoint_identity'] = identity
    save_json(model_path/'decision_threshold.json', policy)
    save_json(model_path/'training_config.json', read_json(source_run/'config.json'))
    load_decision_policy(model_path)
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--source-run', type=Path)
    args = parser.parse_args()
    settings = read_json(ROOT/'config/report.json')
    source = args.source_run or ROOT/'outputs/runs'/settings['runs']['frozen_trial']
    print('Prepared checkpoint:', prepare_checkpoint(args.model_path, source))
