import os

from lavis.common.registry import registry
from omegaconf import OmegaConf

from lavis.models import *  # noqa  # isort:skip
from nice.datasets.builders import *  # noqa  # isort:skip
from nice.runners import RunnerBase  # noqa  # isort:skip
from nice.models.blip2_models.blip2_opt import Blip2OPT  # noqa  # isort:skip
from nice.models.blip2_models.blip2_qformer import Blip2Qformer  # noqa  # isort:skip
from nice.common.optims import LinearWarmupCosineLRScheduler  # noqa  # isort:skip
from lavis.common.optims import LinearWarmupStepLRScheduler  # noqa  # isort:skip


def register_path(registry, name, path):
    if name in registry.mapping["paths"]:
        # first unregister the key with name
        registry.mapping["paths"].pop(name, None)
    registry.register_path(name, path)


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

register_path(registry, "library_root", root_dir)

repo_root = os.path.join(root_dir, "..")
register_path(registry, "repo_root", repo_root)

cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
register_path(registry, "cache_root", cache_root)

# registry for NICE datasets
nice_cfg = OmegaConf.load(os.path.join(root_dir, "configs/datasets/nice/defaults.yaml"))
register_path(registry, "nice_root", nice_cfg.datasets.nice.data_dir)

# registry for shutterstock datasets
shutterstock_cfg = OmegaConf.load(
    os.path.join(root_dir, "configs/datasets/shutterstock/defaults.yaml")
)
register_path(registry, "shutterstock_root", shutterstock_cfg.datasets.shutterstock.data_dir)
