import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for all randomness sources to achieve deterministic behavior.

    This function seeds Python's random module, NumPy's random generator,
    and PyTorch (CPU + CUDA). It also enables cuDNN deterministic mode
    and disables cuDNN benchmarking, which ensures that the same input
    always produces the same output on GPU.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
