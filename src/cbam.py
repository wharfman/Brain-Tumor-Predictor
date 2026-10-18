"""Convolutional Block Attention Module (CBAM) for the YOLO classification backbone.

CBAM (Woo et al., 2018, https://arxiv.org/abs/1807.06521) refines a feature map in two steps:

1. Channel attention: global average and max pooling each give a C-length descriptor, both pass
   through a shared two-layer MLP (C -> C/r -> C, ReLU between), are summed and squashed with a
   sigmoid. The input is multiplied by these per-channel weights.
2. Spatial attention: the channel-refined map is pooled across channels (mean and max), the two
   H x W maps go through a 7x7 convolution and a sigmoid, and the result multiplies every channel.

Ultralytics ships its own `CBAM`, but its channel attention uses only average pooling and a single
full-width 1x1 conv, so this module implements the paper's version.

Ultralytics' trainer rebuilds the network from its YAML before training, so CBAM is attached by
`CBAMClassificationTrainer` after the pretrained weights are loaded (see config.CBAM_LAYERS).
"""

import torch
from torch import nn
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.utils import RANK

from src import config


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        # Shared MLP as 1x1 convs, applied to both pooled descriptors.
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean((2, 3), keepdim=True))
        peak = self.mlp(x.amax((2, 3), keepdim=True))
        return torch.sigmoid(avg + peak)  # B x C x 1 x 1


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return torch.sigmoid(self.conv(pooled))  # B x 1 x H x W


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        return x * self.spatial(x)


class CBAMBlock(nn.Module):
    """A backbone layer followed by CBAM, standing in for that layer in Ultralytics' layer list."""

    def __init__(self, block: nn.Module, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.block = block
        self.cbam = CBAM(channels, reduction, kernel_size)
        # Ultralytics routes each layer's input/output by these attributes (see BaseModel._predict_once).
        self.f, self.i, self.type = block.f, block.i, f"{block.type}+CBAM"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.block(x))


def has_cbam(model: nn.Module) -> bool:
    return any(isinstance(m, CBAMBlock) for m in model.modules())


def add_cbam(model: nn.Module, layers=None, reduction: int | None = None, kernel_size: int | None = None) -> nn.Module:
    """Wrap the given layer indices of an Ultralytics model (in place) with CBAM; returns the model.

    Layers that are already wrapped are left alone. Defaults come from config.
    """
    layers = config.CBAM_LAYERS if layers is None else layers
    reduction = reduction or config.CBAM_REDUCTION
    kernel_size = kernel_size or config.CBAM_KERNEL
    if not layers:
        return model
    # Output channels per layer, from a dry run, since Ultralytics blocks don't expose them uniformly.
    first = next(model.parameters())
    channels = {}

    def record(i):
        def hook(_module, _inputs, output):  # must return None, or it would replace the layer's output
            channels[i] = output.shape[1]

        return hook

    hooks = [model.model[i].register_forward_hook(record(i)) for i in layers]
    was_training = model.training
    try:
        with torch.no_grad():
            model.eval()(torch.zeros(1, model.yaml.get("channels", 3), 64, 64, device=first.device, dtype=first.dtype))
    finally:
        model.train(was_training)
        for hook in hooks:
            hook.remove()
    for i in layers:
        if not isinstance(model.model[i], CBAMBlock):
            model.model[i] = CBAMBlock(model.model[i], channels[i], reduction, kernel_size).to(first.device)
    return model


class CBAMClassificationTrainer(ClassificationTrainer):
    """Ultralytics' classification trainer, with CBAM added to the backbone (config.CBAM_LAYERS)."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None and has_cbam(weights):
            # Starting from a CBAM checkpoint: add CBAM first so its weights load too.
            model = add_cbam(super().get_model(cfg, None, verbose=False))
            model.load(weights)
        else:
            # Starting from a plain checkpoint: load it first, while layer names still match.
            model = add_cbam(super().get_model(cfg, weights, verbose=False))
        if verbose and RANK == -1:
            model.info()
        return model
