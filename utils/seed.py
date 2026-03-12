import os
import random
import numpy as np

def seed_everything(seed: int) -> int:
    """
    Implement seeding to ensure reproducibility.
    The seed is used inside Math operations to ensure a consistent sequence of random numbers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    print(f"[seed] seed={seed} mode=best-effort")
    return seed