import pytest
import torch
from ultralytics import YOLO

from src import config
from src.bifpn import BiFPN, BiFPNClassify, BiFPNLayer, FusionNode, add_bifpn, has_bifpn
from src.cbam import has_cbam
from src.sppf_plus import has_sppf_plus
from src.trainer import CustomClassificationTrainer


@pytest.fixture
def plain_model():
    """An untrained YOLO11n classifier built from YAML (no download)."""
    return YOLO("yolo11n-cls.yaml").model


def test_fusion_weights_are_normalized():
    node = FusionNode(4, 2)
    node.conv = torch.nn.Identity()  # look at the fusion alone
    with torch.no_grad():
        node.weights.copy_(torch.tensor([3.0, -1.0]))  # negative weights are clipped to 0 by ReLU
    a, b = torch.ones(1, 4, 2, 2), torch.full((1, 4, 2, 2), 5.0)
    assert torch.allclose(node([a, b]), a, atol=1e-3)


def test_bifpn_keeps_level_shapes():
    feats = [torch.randn(2, 16, 16, 16), torch.randn(2, 24, 8, 8), torch.randn(2, 32, 4, 4)]
    out = BiFPN([16, 24, 32], channels=8, repeats=3)(feats)
    assert [o.shape for o in out] == [(2, 8, 16, 16), (2, 8, 8, 8), (2, 8, 4, 4)]
    assert len(BiFPN([16, 24, 32], 8, repeats=3).layers) == 3


def test_bifpn_layer_has_extra_p4_edge():
    layer = BiFPNLayer(8)
    assert len(layer.p4_out.weights) == 3  # original P4, top-down P4, downsampled P3
    feats = [torch.randn(1, 8, 10, 10), torch.randn(1, 8, 5, 5), torch.randn(1, 8, 3, 3)]  # odd sizes
    assert [o.shape[-2:] for o in layer(feats)] == [f.shape[-2:] for f in feats]


def test_add_bifpn_replaces_head(plain_model):
    before = {k: v.clone() for k, v in plain_model.state_dict().items() if not k.startswith("model.10.")}
    model = add_bifpn(plain_model, layers=(4, 6, 9), repeats=2)
    head = model.model[-1]
    assert isinstance(head, BiFPNClassify) and head.f == [4, 6, 9] and head.i == 10
    assert {4, 6, 9} <= set(model.save)
    assert [c.conv.in_channels for c in head.bifpn.lateral] == [128, 128, 256]  # yolo11n widths
    after = model.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items())  # backbone untouched

    x = torch.rand(2, 3, 64, 64)
    model.train()
    assert model(x).shape == (2, model.yaml["nc"])  # logits while training, like Classify
    model.eval()
    probs, logits = model(x)
    assert torch.allclose(probs, logits.softmax(1)) and torch.allclose(probs.sum(1), torch.ones(2))

    add_bifpn(model, layers=(4, 6, 9))  # already replaced: unchanged
    assert model.model[-1] is head


def test_add_bifpn_disabled(plain_model):
    assert not has_bifpn(add_bifpn(plain_model, layers=()))


def _trainer_model(weights, monkeypatch):
    monkeypatch.setattr(config, "CBAM_LAYERS", (4, 6, 8))
    monkeypatch.setattr(config, "SPPF_PLUS_LAYER", 9)
    monkeypatch.setattr(config, "BIFPN_LAYERS", (4, 6, 9))
    trainer = CustomClassificationTrainer.__new__(CustomClassificationTrainer)  # skip dataset setup
    trainer.data = {"nc": len(config.CLASS_NAMES), "channels": 3}
    trainer.args = type("Args", (), {"pretrained": True, "dropout": 0.0})()
    trainer.resume = False
    return trainer.get_model(cfg=weights.yaml, weights=weights, verbose=False)


def test_trainer_adds_all_modules(plain_model, monkeypatch):
    model = _trainer_model(plain_model, monkeypatch)
    assert has_cbam(model) and has_sppf_plus(model) and has_bifpn(model)
    assert model.model[-1].nc == len(config.CLASS_NAMES)
    model.eval()
    probs, _ = model(torch.rand(1, 3, 64, 64))
    assert probs.shape == (1, len(config.CLASS_NAMES))


def test_trainer_restores_bifpn_weights(plain_model, monkeypatch):
    trained = add_bifpn(plain_model, layers=(4, 6, 9))  # e.g. a checkpoint with BiFPN only
    with torch.no_grad():
        trained.model[-1].bifpn.layers[1].p4_out.weights.copy_(torch.tensor([0.2, 0.3, 0.5]))
    model = _trainer_model(trained, monkeypatch)
    assert torch.equal(model.model[-1].bifpn.layers[1].p4_out.weights, trained.model[-1].bifpn.layers[1].p4_out.weights)
    assert has_cbam(model) and has_sppf_plus(model)


def test_bifpn_checkpoint_round_trip(plain_model, tmp_path):
    model = add_bifpn(plain_model, layers=(4, 6, 9)).eval()
    path = tmp_path / "bifpn.pt"
    torch.save({"model": model, "train_args": {"task": "classify"}}, path)

    loaded = YOLO(str(path))
    assert loaded.task == "classify" and has_bifpn(loaded.model)
    x = torch.rand(1, 3, 64, 64)
    assert torch.allclose(loaded.model.float().eval()(x)[0], model(x)[0], atol=1e-6)
