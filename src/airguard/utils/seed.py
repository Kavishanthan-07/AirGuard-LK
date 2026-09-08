import os
import random
import numpy as np

SEED = 42

def set_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    return seed
