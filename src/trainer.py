"""Classification trainer that adds this project's modules (CBAM, SPPF+, BiFPN head) to the model.

Ultralytics' trainer rebuilds the network from its YAML before training, so modules can't be added to
the `YOLO` object beforehand; they are added here, after the pretrained weights are loaded.
"""

from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.utils import RANK

from src.bifpn import add_bifpn, has_bifpn
from src.cbam import add_cbam, has_cbam
from src.sppf_plus import add_sppf_plus, has_sppf_plus

# (detect, add) pairs; each add reads its layers from config, is a no-op when disabled and skips
# layers that are already wrapped.
MODULES = ((has_cbam, add_cbam), (has_sppf_plus, add_sppf_plus), (has_bifpn, add_bifpn))


class CustomClassificationTrainer(ClassificationTrainer):
    """Ultralytics' classification trainer, with CBAM (config.CBAM_LAYERS) and SPPF+
    (config.SPPF_PLUS_LAYER) added to the backbone, and a BiFPN head (config.BIFPN_LAYERS)."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        present = [add for has, add in MODULES if weights is not None and has(weights)]
        if present:
            # The starting weights already contain some modules: add those first so their weights load too.
            model = super().get_model(cfg, None, verbose=False)
            for add in present:
                add(model)
            model.load(weights)
        else:
            # Plain checkpoint: load it first, while layer names still match.
            model = super().get_model(cfg, weights, verbose=False)
        for _, add in MODULES:
            add(model)
        if verbose and RANK == -1:
            model.info()
        return model
