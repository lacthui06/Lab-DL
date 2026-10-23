"""Model creation and inference helpers for Lab 03."""
from typing import Dict, Any, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase
)


def get_sentiment_pipeline(model_name: str) -> Pipeline:
    """Create a high-level sentiment analysis pipeline."""
    return pipeline("sentiment-analysis", model=model_name)


def load_classification_model(
    model_name: str,
    num_labels: int = 2,
    id2label: Dict[int, str] = None,
    label2id: Dict[str, int] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer for sequence classification."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model, tokenizer


def predict_sentiment_manual(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Tokenize and run manual forward pass to compute softmax probabilities."""
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        pred_idx = int(torch.argmax(logits, dim=-1).item())

    label = model.config.id2label.get(pred_idx, str(pred_idx))
    confidence = probs[pred_idx] if isinstance(probs, list) else probs

    return {
        "text": text,
        "predicted_label": label,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": probs,
        "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    }
