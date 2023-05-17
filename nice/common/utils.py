import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lavis.common.dist_utils import get_rank


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
