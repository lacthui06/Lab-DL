"""Deterministic per-image features extracted before pixel standardization."""
import numpy as np

FEATURE_NAMES = ("mean", "std", "foreground_ratio", "bbox_width", "bbox_height",
                 "aspect_ratio", "centroid_x", "centroid_y", "symmetry", "edge_density")


def foreground_bbox(mask):
    """Enclosing height/width for N x H x W masks; empty foreground maps to zero."""
    rows, cols = mask.any(axis=2), mask.any(axis=1)
    nonempty = rows.any(axis=1)
    height = np.where(nonempty, rows.shape[1]-rows[:, ::-1].argmax(axis=1)-rows.argmax(axis=1),0)
    width = np.where(nonempty, cols.shape[1]-cols[:, ::-1].argmax(axis=1)-cols.argmax(axis=1),0)
    return height, width


def measure_intensity(image, threshold=0.1):
    """Shared intensity/edge definition for arbitrary H x W images."""
    edges = np.concatenate((np.abs(np.diff(image, axis=0)).ravel(),
                            np.abs(np.diff(image, axis=1)).ravel()))
    density = (edges > threshold).mean() if edges.size else 0.0
    return image.mean(), image.std(), density


def extract_features(images, threshold=0.1):
    """Input: finite N x 28 x 28 float images in [0,1]; empty shapes map to zero."""
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (28, 28):
        raise ValueError("Expected N x 28 x 28")
    if not np.isfinite(x).all() or x.min(initial=0) < 0 or x.max(initial=0) > 1:
        raise ValueError("Features require finite rescaled pixels in [0,1]")
    result = np.zeros((len(x), 10), dtype=np.float32)
    yy, xx = np.indices((28, 28))
    heights, widths = foreground_bbox(x > threshold)
    for i, image in enumerate(x):
        mask = image > threshold
        width, height = widths[i], heights[i]
        mass = image.sum()
        mean, std, edge_density = measure_intensity(image, threshold)
        result[i] = (mean, std, mask.mean(), width/28, height/28,
                     width/height if height else 0,
                     (image*xx).sum()/mass/27 if mass else 0,
                     (image*yy).sum()/mass/27 if mass else 0,
                     1-np.abs(image-image[:, ::-1]).mean(), edge_density)
    return result
