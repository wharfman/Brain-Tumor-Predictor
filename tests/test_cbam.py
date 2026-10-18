import pytest
import torch
from ultralytics import YOLO

from src import config
from src.cbam import CBAM, CBAMBlock, ChannelAttention, CBAMClassificationTrainer, SpatialAttention, add_cbam, has_cbam


def probabilities(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out  # eval-mode Classify returns (probabilities, logits)


@pytest.fixture
def plain_model():
    """An untrained YOLO11n classifier built from YAML (no download)."""
    return YOLO("yolo11n-cls.yaml").model


def test_attention_shapes_and_ranges():
    x = torch.randn(2, 32, 9, 7)
    channel = ChannelAttention(32, reduction=16)
    assert channel.mlp[0].out_channels == 2  # C / r hidden units
    weights = channel(x)
    assert weights.shape == (2, 32, 1, 1) and ((weights > 0) & (weights < 1)).all()

    spatial = SpatialAttention(7)(x)
    assert spatial.shape == (2, 1, 9, 7) and ((spatial > 0) & (spatial < 1)).all()

    assert CBAM(32)(x).shape == x.shape


def test_channel_attention_uses_average_and_max_pooling():
    channel = ChannelAttention(4, reduction=2)
    for conv in (channel.mlp[0], channel.mlp[2]):  # positive weights, so ReLU can't zero both descriptors
        torch.nn.init.constant_(conv.weight, 0.1)
    # Same mean per channel, different maxima: an average-only module couldn't tell these apart.
    flat = torch.ones(1, 4, 2, 2)
    peaked = torch.tensor([4.0, 0.0, 0.0, 0.0]).view(1, 1, 2, 2).expand(1, 4, 2, 2)
    assert not torch.allclose(channel(flat), channel(peaked))


def test_add_cbam_wraps_layers_and_keeps_weights(plain_model):
    before = {k: v.clone() for k, v in plain_model.state_dict().items()}
    model = add_cbam(plain_model, layers=(4, 6, 8))

    assert [i for i, m in enumerate(model.model) if isinstance(m, CBAMBlock)] == [4, 6, 8]
    for i in (4, 6, 8):
        block = model.model[i]
        assert (block.f, block.i) == (block.block.f, block.block.i)
    after = model.state_dict()
    for key, value in before.items():  # every original tensor is still there, just nested in .block
        for i in (4, 6, 8):
            key = key.replace(f"model.{i}.", f"model.{i}.block.", 1)
        assert torch.equal(after[key], value)

    model.eval()
    assert probabilities(model, torch.zeros(1, 3, 64, 64)).shape == (1, model.yaml["nc"])

    add_cbam(model, layers=(4, 6, 8))  # already wrapped: unchanged
    assert not any(isinstance(m.block, CBAMBlock) for m in model.model if isinstance(m, CBAMBlock))


def test_add_cbam_disabled(plain_model):
    assert not has_cbam(add_cbam(plain_model, layers=()))


def _trainer_model(weights, monkeypatch):
    monkeypatch.setattr(config, "CBAM_LAYERS", (4, 6, 8))
    trainer = CBAMClassificationTrainer.__new__(CBAMClassificationTrainer)  # skip dataset setup
    trainer.data = {"nc": len(config.CLASS_NAMES), "channels": 3}
    trainer.args = type("Args", (), {"pretrained": True, "dropout": 0.0})()
    trainer.resume = False
    return trainer.get_model(cfg=weights.yaml, weights=weights, verbose=False)


def test_trainer_loads_plain_weights_then_adds_cbam(plain_model, monkeypatch):
    model = _trainer_model(plain_model, monkeypatch)
    assert has_cbam(model)
    stem = "model.0.conv.weight"
    assert torch.equal(model.state_dict()[stem], plain_model.state_dict()[stem])
    deep = "model.8.cv1.conv.weight"  # inside a wrapped layer
    assert torch.equal(model.state_dict()[deep.replace("model.8.", "model.8.block.")], plain_model.state_dict()[deep])


def test_trainer_restores_cbam_weights(plain_model, monkeypatch):
    trained = add_cbam(plain_model, layers=(4, 6, 8))
    with torch.no_grad():
        trained.model[6].cbam.spatial.conv.weight.fill_(0.123)
    model = _trainer_model(trained, monkeypatch)
    assert torch.equal(model.model[6].cbam.spatial.conv.weight, trained.model[6].cbam.spatial.conv.weight)


def test_cbam_checkpoint_round_trip(plain_model, tmp_path):
    model = add_cbam(plain_model, layers=(4, 6, 8)).eval()
    path = tmp_path / "cbam.pt"
    torch.save({"model": model, "train_args": {"task": "classify"}}, path)

    loaded = YOLO(str(path))
    assert loaded.task == "classify" and has_cbam(loaded.model)
    x = torch.rand(1, 3, 64, 64)
    assert torch.allclose(probabilities(loaded.model.float().eval(), x), probabilities(model, x))
