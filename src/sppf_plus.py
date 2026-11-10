"""SPPF+ multi-scale pooling block for the YOLO classification backbone.

SPPF+ combines the feature reuse of SPPF (each max pool runs on the previous pool's output, so the
receptive field grows without pooling the input again) with the cross-stage partial layout of
SPPCSPC (a pooled branch and a 1x1 shortcut branch, concatenated and fused):

    pooled:   x -> cv1 (1x1) -> cv3 (3x3) -> cv4 (1x1) = a
              p1 = pool5(a), p2 = pool9(p1), p3 = pool13(p2)      (stride 1, same size)
              cat(a, p1, p2, p3) -> cv5 (1x1) -> cv6 (3x3)
    shortcut: x -> cv2 (1x1)
    out:      cat(pooled, shortcut) -> cv7 (1x1)

Chained pools of 5, 9 and 13 see 5x5, 13x13 and 25x25 windows, so on the 20x20 map at the end of
the backbone (640 input) the last one covers the whole image: fine detail and global context are
fused before the classifier.

YOLO11's classification backbone has no SPPF (the detection model puts it right before C2PSA), so
SPPF+ is inserted in front of config.SPPF_PLUS_LAYER by `SPPFPlusBlock`. Like CBAM, it is attached
after the pretrained weights are loaded (see src/trainer.py).
"""

import torch
from torch import nn
from ultralytics.nn.modules import Conv

from src import config


class SPPFPlus(nn.Module):
    def __init__(self, c1: int, c2: int, kernels=(5, 9, 13), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(c_, c_, 3)
        self.cv4 = Conv(c_, c_, 1)
        self.pools = nn.ModuleList(nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernels)
        self.cv5 = Conv(c_ * (len(kernels) + 1), c_, 1)
        self.cv6 = Conv(c_, c_, 3)
        self.cv7 = Conv(2 * c_, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv4(self.cv3(self.cv1(x)))]
        for pool in self.pools:
            y.append(pool(y[-1]))  # reuse the previous pool's output (SPPF)
        pooled = self.cv6(self.cv5(torch.cat(y, 1)))
        return self.cv7(torch.cat((pooled, self.cv2(x)), 1))


class SPPFPlusBlock(nn.Module):
    """SPPF+ followed by a backbone layer, standing in for that layer in Ultralytics' layer list.

    SPPF+ is added residually and its output BN starts at zero, so a freshly inserted block passes
    the pretrained features through unchanged and learns its contribution during training.
    """

    def __init__(self, block: nn.Module, channels: int, kernels=(5, 9, 13)):
        super().__init__()
        self.sppf = SPPFPlus(channels, channels, kernels)
        nn.init.zeros_(self.sppf.cv7.bn.weight)
        self.block = block
        # Ultralytics routes each layer's input/output by these attributes (see BaseModel._predict_once).
        self.f, self.i, self.type = block.f, block.i, f"SPPF+>{block.type}"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x + self.sppf(x))


def has_sppf_plus(model: nn.Module) -> bool:
    return any(isinstance(m, SPPFPlusBlock) for m in model.modules())


def add_sppf_plus(model: nn.Module, layer=None, kernels=None) -> nn.Module:
    """Insert SPPF+ in front of the given layer index of an Ultralytics model (in place); returns it.

    A layer that is already wrapped is left alone; layer None (the default) uses config, and
    config.SPPF_PLUS_LAYER = None turns SPPF+ off.
    """
    layer = config.SPPF_PLUS_LAYER if layer is None else layer
    kernels = kernels or config.SPPF_PLUS_KERNELS
    if layer is None or isinstance(model.model[layer], SPPFPlusBlock):
        return model
    block = model.model[layer]
    # Ultralytics blocks start with a convolution of their input, which gives the input channels.
    channels = next(m for m in block.modules() if isinstance(m, nn.Conv2d)).in_channels
    first = next(model.parameters())
    model.model[layer] = SPPFPlusBlock(block, channels, kernels).to(first.device)
    return model
