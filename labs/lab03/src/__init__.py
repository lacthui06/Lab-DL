"""Init for lab03 src modules."""
from .datasets.data_loader import load_imdb_dataset, tokenize_dataset, run_text_eda
from .models.classifier import get_sentiment_pipeline, load_classification_model, predict_sentiment_manual
from .training.trainer import build_trainer, run_one_batch_sanity_check
from .utils.metrics import get_compute_metrics_fn, plot_confusion_matrix, save_environment_snapshot

__all__ = [
    "load_imdb_dataset",
    "tokenize_dataset",
    "run_text_eda",
    "get_sentiment_pipeline",
    "load_classification_model",
    "predict_sentiment_manual",
    "build_trainer",
    "run_one_batch_sanity_check",
    "get_compute_metrics_fn",
    "plot_confusion_matrix",
    "save_environment_snapshot",
]
