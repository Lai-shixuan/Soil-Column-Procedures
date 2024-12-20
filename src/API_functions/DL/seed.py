import torch
import numpy as np
import random


def stablize_seed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)    # to ensure reproducibility
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)