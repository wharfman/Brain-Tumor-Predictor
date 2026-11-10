import pytest
import torch
from torch import nn
from ultralytics import YOLO

from src import config
from src.cbam import add_cbam, has_cbam
from src.sppf_plus import SPPFPlus, SPPFPlusBlock, add_sppf_plus, has_sppf_plus
from src.trainer import CustomClassificationTrainer


def probabilities(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out  # eval-mode Classify returns (probabilities, logits)


@pytest.fixture
def plain_model():
    """An untrained YOLO11n classifier built from YAML (no download)."""
    return YOLO("yolo11n-cls.yaml").model


def test_sppf_plus_shape():
    module = SPPFPlus(32, 48)
    assert module(torch.randn(2, 32, 9, 7)).shape == (2, 48, 9, 7)
    assert module.cv5.conv.in_channels == 4 * 24  # input + three pooled maps, hidden width c2 / 2


def test_pools_are_chained():
    module = SPPFPlus(8, 8, kernels=(5, 9, 13))
    pooled = []
    for pool in module.pools:
        pool.register_forward_hook(lambda _m, _i, out: pooled.append(out))
    x = torch.randn(1, 8, 30, 30)
    module.eval()(x)
    a = module.cv4(module.cv3(module.cv1(x)))
    # Each pool runs on the previous one's output, so the windows grow to 5, 13 and 25.
    for out, k in zip(pooled, (5, 13, 25)):
        assert torch.allclose(out, nn.functional.max_pool2d(a, k, stride=1, padding=k // 2))


def test_add_sppf_plus_starts_as_identity_and_keeps_weights(plain_model):
    plain_model.eval()
    x = torch.rand(1, 3, 64, 64)
    expected = probabilities(plain_model, x)
    before = {k: v.clone() for k, v in plain_model.state_dict().items()}

    model = add_sppf_plus(plain_model, layer=9)
    block = model.model[9]
    assert isinstance(block, SPPFPlusBlock) and (block.f, block.i) == (block.block.f, block.block.i)
    assert block.sppf.cv1.conv.in_channels == model.model[8].cv2.conv.out_channels
    after = model.state_dict()
    for key, value in before.items():
        assert torch.equal(after[key.replace("model.9.", "model.9.block.", 1)], value)
    assert torch.allclose(probabilities(model.eval(), x), expected)  # untrained SPPF+ adds zero

    add_sppf_plus(model, layer=9)  # already wrapped: unchanged
    assert not isinstance(model.model[9].block, SPPFPlusBlock)


def test_add_sppf_plus_disabled(plain_model, monkeypatch):
    monkeypatch.setattr(config, "SPPF_PLUS_LAYER", None)
    assert not has_sppf_plus(add_sppf_plus(plain_model))


def _trainer_model(weights, monkeypatch):
    monkeypatch.setattr(config, "CBAM_LAYERS", (4, 6, 8))
    monkeypatch.setattr(config, "SPPF_PLUS_LAYER", 9)
    trainer = CustomClassificationTrainer.__new__(CustomClassificationTrainer)  # skip dataset setup
    trainer.data = {"nc": len(config.CLASS_NAMES), "channels": 3}
    trainer.args = type("Args", (), {"pretrained": True, "dropout": 0.0})()
    trainer.resume = False
    return trainer.get_model(cfg=weights.yaml, weights=weights, verbose=False)


def test_trainer_loads_plain_weights_then_adds_sppf_plus(plain_model, monkeypatch):
    model = _trainer_model(plain_model, monkeypatch)
    assert has_sppf_plus(model) and has_cbam(model)
    key = "model.9.cv1.conv.weight"  # inside the wrapped C2PSA
    assert torch.equal(model.state_dict()[key.replace("model.9.", "model.9.block.")], plain_model.state_dict()[key])


def test_trainer_restores_sppf_plus_weights_and_adds_missing_cbam(plain_model, monkeypatch):
    trained = add_sppf_plus(plain_model, layer=9)  # e.g. a checkpoint trained with SPPF+ but no CBAM
    with torch.no_grad():
        trained.model[9].sppf.cv6.conv.weight.fill_(0.123)
    model = _trainer_model(trained, monkeypatch)
    assert torch.equal(model.model[9].sppf.cv6.conv.weight, trained.model[9].sppf.cv6.conv.weight)
    assert has_cbam(model)


def test_trainer_restores_both(plain_model, monkeypatch):
    trained = add_sppf_plus(add_cbam(plain_model, layers=(4, 6, 8)), layer=9)
    with torch.no_grad():
        trained.model[9].sppf.cv6.conv.weight.fill_(0.123)
        trained.model[6].cbam.spatial.conv.weight.fill_(0.456)
    model = _trainer_model(trained, monkeypatch)
    assert torch.equal(model.model[9].sppf.cv6.conv.weight, trained.model[9].sppf.cv6.conv.weight)
    assert torch.equal(model.model[6].cbam.spatial.conv.weight, trained.model[6].cbam.spatial.conv.weight)


def test_sppf_plus_checkpoint_round_trip(plain_model, tmp_path):
    model = add_sppf_plus(plain_model, layer=9)
    nn.init.ones_(model.model[9].sppf.cv7.bn.weight)  # as if trained, so SPPF+ affects the output
    model.eval()
    path = tmp_path / "sppf_plus.pt"
    torch.save({"model": model, "train_args": {"task": "classify"}}, path)

    loaded = YOLO(str(path))
    assert loaded.task == "classify" and has_sppf_plus(loaded.model)
    x = torch.rand(1, 3, 64, 64)
    assert torch.allclose(probabilities(loaded.model.float().eval(), x), probabilities(model, x), atol=1e-6)
