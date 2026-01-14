import random
import numpy as np
import torch
import os

def set_seed(seed, deterministic=True):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enforce deterministic behavior (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    print(f"Random seed set to {seed} (deterministic={deterministic})")


def get_device(device_str='auto'):
    """
    Get PyTorch device.

    Args:
        device_str: 'auto', 'cuda', 'cpu', or specific like 'cuda:0'

    Returns:
        torch.device
    """
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)
