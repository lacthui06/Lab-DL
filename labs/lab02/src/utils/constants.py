"""Shared constants and metadata for Lab 02: CIFAR-10 transfer learning.

Single source of truth for class labels, integrity checksums, and dataset shapes.
Modeled after Lab 01 src/utils/helpers.py configuration conventions.
"""

CLASSES: list[str] = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

CHECKSUMS: list[str] = [
    'c99cafc152244af753f735de768cd75f',
    'd4bba439e000b95fd0a9bffe97cbabec',
    '54ebc095f3ab1f0389bbae665268c751',
    '634d18415352ddfa80567beed471001a',
    '482c414d41f54cd18b22e5b47cb7c3cb',
    '40351d587109b95175f43aff81a1287e',
]

NUM_CLASSES: int = 10
NUM_CHANNELS: int = 3
DEFAULT_IMAGE_SIZE: int = 224
RANDOM_SEED: int = 42

__all__ = [
    "CLASSES",
    "CHECKSUMS",
    "NUM_CLASSES",
    "NUM_CHANNELS",
    "DEFAULT_IMAGE_SIZE",
    "RANDOM_SEED",
]
