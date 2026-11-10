"""Bidirectional Feature Pyramid Network (BiFPN) head for the YOLO classifier.

BiFPN (Tan et al., EfficientDet, 2020, https://arxiv.org/abs/1911.09070) fuses backbone features of
three sizes (P3, P4, P5) in both directions, and repeats that layer several times:

    top-down:  P4_td  = node(P4, up(P5))
               P3_out = node(P3, up(P4_td))
    bottom-up: P4_out = node(P4, P4_td, down(P3_out))   <- extra edge from the original P4
               P5_out = node(P5, down(P4_out))

Each node is a weighted sum of its inputs, with learnable non-negative weights normalised to sum to
1 ("fast normalized fusion"), followed by a depthwise-separable 3x3 convolution. Fine detail from P3
reaches P5, and semantics from P5 reach P3.

YOLO11's classification head only sees the last backbone map, so `BiFPNClassify` replaces it: it
takes config.BIFPN_LAYERS (80x80, 40x40, 20x20 for a 640 input) through the BiFPN, average-pools
each output and classifies the concatenation. Like CBAM, it is added after the pretrained weights
are loaded (see src/trainer.py).
"""

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules import Conv, DWConv

from src import config


class FusionNode(nn.Module):
    """Fast normalized fusion of same-shaped inputs, then a depthwise-separable 3x3 conv."""

    def __init__(self, channels: int, inputs: int, eps: float = 1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(inputs))
        self.eps = eps
        self.conv = nn.Sequential(DWConv(channels, channels, 3), Conv(channels, channels, 1))

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        return self.conv(sum(wi * x for wi, x in zip(w, xs)))


def _up(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=like.shape[-2:], mode="nearest")


def _down(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=2, padding=1)  # ceil(n / 2), like the backbone's stride-2 convs


class BiFPNLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.p4_td = FusionNode(channels, 2)
        self.p3_out = FusionNode(channels, 2)
        self.p4_out = FusionNode(channels, 3)
        self.p5_out = FusionNode(channels, 2)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = feats
        p4_td = self.p4_td([p4, _up(p5, p4)])
        p3_out = self.p3_out([p3, _up(p4_td, p3)])
        p4_out = self.p4_out([p4, p4_td, _down(p3_out)])
        p5_out = self.p5_out([p5, _down(p4_out)])
        return [p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    """1x1 convs bring each level to a common width, then `repeats` BiFPN layers."""

    def __init__(self, in_channels, channels: int, repeats: int = 2):
        super().__init__()
        self.lateral = nn.ModuleList(Conv(c, channels, 1) for c in in_channels)
        self.layers = nn.Sequential(*(BiFPNLayer(channels) for _ in range(repeats)))

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.layers([conv(x) for conv, x in zip(self.lateral, feats)])


class BiFPNClassify(nn.Module):
    """Replaces Ultralytics' Classify head: BiFPN over several backbone layers, pool, concat, linear.

    Returns logits in training, and (probabilities, logits) otherwise, like Classify.
    """

    export = False  # Ultralytics sets this when exporting; then only probabilities are returned

    def __init__(self, head: nn.Module, layers, in_channels, repeats: int = 2):
        super().__init__()
        channels = min(in_channels)
        self.bifpn = BiFPN(in_channels, channels, repeats)
        self.nc = head.linear.out_features
        self.drop = nn.Dropout(head.drop.p)
        self.linear = nn.Linear(channels * len(in_channels), self.nc)
        # Ultralytics routes each layer's input/output by these attributes (see BaseModel._predict_once).
        self.f, self.i, self.type = list(layers), head.i, "BiFPNClassify"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor | tuple:
        pooled = [F.adaptive_avg_pool2d(x, 1).flatten(1) for x in self.bifpn(feats)]
        x = self.linear(self.drop(torch.cat(pooled, 1)))
        if self.training:
            return x
        y = x.softmax(1)
        return y if self.export else (y, x)


def has_bifpn(model: nn.Module) -> bool:
    return any(isinstance(m, BiFPNClassify) for m in model.modules())


def add_bifpn(model: nn.Module, layers=None, repeats: int | None = None) -> nn.Module:
    """Replace the Classify head of an Ultralytics model (in place) with a BiFPN head; returns the model.

    A model that already has one is left alone. Defaults come from config; config.BIFPN_LAYERS = ()
    turns BiFPN off.
    """
    layers = config.BIFPN_LAYERS if layers is None else layers
    repeats = repeats or config.BIFPN_REPEATS
    if not layers or has_bifpn(model):
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
    head = BiFPNClassify(model.model[-1], layers, [channels[i] for i in layers], repeats)
    model.model[-1] = head.to(device=first.device, dtype=first.dtype)
    model.save = sorted(set(model.save) | set(layers))  # keep these layers' outputs for the head
    return model
