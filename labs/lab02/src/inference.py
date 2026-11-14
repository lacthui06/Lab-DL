"""Shared batch-one inference; checkpoint loading is separate from a request."""
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class LoadedModel:
    """A resident eval model and its checkpoint-specific preprocessing."""
    model: Any
    name: str
    image_size: int
    device: Any
    transform: Callable


def load_model(checkpoint: Path, device: str = 'cpu', threads: int = 4) -> LoadedModel:
    """Load once, without pretrained downloads; fail explicitly when CUDA is unavailable."""
    import torch
    from .datasets.data import image_transform
    from .models.models import build_model
    if threads < 1:
        raise ValueError('threads must be positive')
    target = torch.device(device)
    if target.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA inference are supported')
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing to substitute a CPU benchmark')
    torch.set_num_threads(threads)
    saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    if metadata.get('dataset') != 'CIFAR10':
        raise ValueError('This inference pipeline requires a CIFAR10 checkpoint')
    name, image_size = metadata['model'], metadata['image_size']
    model = build_model(name, pretrained=False, dropout=metadata.get('dropout', 0.))
    model.load_state_dict(saved['state_dict'])
    model = model.float().to(target).eval()
    return LoadedModel(model, name, image_size, target, image_transform(name, image_size))


def predict_image(session: LoadedModel, image: Any, actual: int | None = None,
                  stages: Callable | None = None) -> dict:
    """Read an image/path, transform to (1,3,H,W), and return the existing JSON-ready schema.

    Optional stage contexts are used only by the separate diagnostic timing pass.
    The caller serializes the result; checkpoint and dataset loading do not occur here.
    """
    import torch
    from PIL import Image
    from .utils.constants import CLASSES
    stage = stages if stages is not None else lambda name: nullcontext()
    with torch.inference_mode():
        with stage('read'):
            if isinstance(image, Image.Image):
                rgb = image.convert('RGB')
            else:
                with Image.open(image) as source:
                    rgb = source.convert('RGB')
                    rgb.load()
        with stage('preprocess'):
            tensor = session.transform(rgb).unsqueeze(0)
        with stage('transfer'):
            tensor = tensor.to(session.device)
        with stage('forward'):
            logits = session.model(tensor)
        with stage('postprocess'):
            probabilities = logits.softmax(1)[0].cpu()
            prediction = int(probabilities.argmax())
            return dict(model=session.name, predicted=CLASSES[prediction],
                        probability=float(probabilities[prediction]),
                        actual=CLASSES[actual] if actual is not None else None,
                        probabilities=dict(zip(CLASSES, probabilities.tolist())))
