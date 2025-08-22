from typing import Optional
import random
import numpy as np
import torch


def set_seed(seed: Optional[int] = None, deterministic: bool = False, verbose: bool = False) -> int:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (Optional[int]): The seed to use. If None, a random seed is generated.
        deterministic (bool): Whether to use deterministic algorithms.
        verbose (bool): If True, prints the seed value. Defaults to True.
    """

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        if verbose:
            print(f"Using random seed: {seed}")
    else:
        if verbose:
            print(f"Using specified seed: {seed}")

    # Set the random seed for various libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    return seed