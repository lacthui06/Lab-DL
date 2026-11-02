"""Đóng băng DistilBERT và tạo AdamW có LR riêng cho encoder/head."""
import math
import torch


def configure_frozen_optimizer(model, config):
    count = config["freeze_layers"]
    layers = model.distilbert.transformer.layer
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count < len(layers):
        raise ValueError("freeze_layers must leave at least one trainable encoder layer")
    for key in ("learning_rate", "head_learning_rate"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Learning rates must be finite and positive")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for module in [model.distilbert.embeddings, *list(layers[:count])]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    groups = []
    for encoder in (True, False):
        for decay in (True, False):
            names, parameters = [], []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or name.startswith("distilbert.") != encoder:
                    continue
                # Bias và LayerNorm không decay; trọng số linear/embedding là tensor >=2D.
                if (parameter.ndim >= 2) != decay:
                    continue
                names.append(name)
                parameters.append(parameter)
            if parameters:
                groups.append({"params": parameters,
                    "lr": config["learning_rate"] if encoder else config["head_learning_rate"],
                    "weight_decay": config["weight_decay"] if decay else 0.0,
                    "parameter_names": names})
    optimizer = torch.optim.AdamW(groups, eps=1e-8)
    evidence = {"frozen_embeddings": True, "frozen_encoder_layers": count,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_groups": [{k: v for k, v in group.items() if k != "params"} for group in groups]}
    return optimizer, evidence
