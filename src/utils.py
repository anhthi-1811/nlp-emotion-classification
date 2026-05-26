import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set random seed to ensure reproducibility."""
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch (CPU)
    torch.manual_seed(seed)
    # Set seed for PyTorch (GPU - if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   
    # Ensure deterministic operations for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")
