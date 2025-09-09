import os
import random

import numpy as np
import torch

def set_seed(SEED:int = 42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        ".."
    )
