"""Explicit ImageNet weights, classifier replacement, and freeze policies."""
import torch
from torch import nn
from torchvision import models

SPECS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
}


def build_model(name, pretrained=True):
    builder, weights = SPECS[name]
    model = builder(weights=weights if pretrained else None)
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    return model


def classifier(model, name):
    return model.fc if name == "resnet18" else model.classifier


def features(model, name, images):
    if name == "resnet18":
        x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        return torch.flatten(model.avgpool(x), 1)
    x = torch.relu(model.features(images))
    return torch.flatten(nn.functional.adaptive_avg_pool2d(x, (1, 1)), 1)


def configure_trainable(model, name, stage):
    for p in model.parameters():
        p.requires_grad = False
    for p in classifier(model, name).parameters():
        p.requires_grad = True
    if stage == "finetune":
        blocks = [model.layer4] if name == "resnet18" else [
            model.features.denseblock4, model.features.norm5]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = True


def training_mode(model):
    # Keep all BatchNorm running statistics fixed, including in unfrozen blocks.
    # requires_grad independently controls gradients for unfrozen affine weights.
    model.train()
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def last_block(model, name):
    return model.layer4 if name == 'resnet18' else model.features.denseblock4


class CachedTail(nn.Module):
    """Train the identical last block/head from cached, frozen-prefix activations.

    Valid for these two architectures with deterministic transforms and fixed BN
    statistics. Checkpoints retain the ORIGINAL full-model state_dict schema.
    """
    def __init__(self, model, name):
        super().__init__()
        self.model, self.name = model, name

    def forward(self, prefix):
        if self.name == 'resnet18':
            x = self.model.layer4(prefix)
            return self.model.fc(torch.flatten(self.model.avgpool(x), 1))
        x = self.model.features.norm5(self.model.features.denseblock4(prefix))
        x = nn.functional.adaptive_avg_pool2d(torch.relu(x), (1, 1))
        return self.model.classifier(torch.flatten(x, 1))

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
