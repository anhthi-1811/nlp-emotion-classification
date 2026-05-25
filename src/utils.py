import random
import numpy as np
import torch

def set_seed(seed=42):
    """Cố định random seed để đảm bảo tính tái lập (reproducibility)"""
    # Cố định cho Python
    random.seed(seed)
    # Cố định cho Numpy
    np.random.seed(seed)
    # Cố định cho PyTorch (CPU)
    torch.manual_seed(seed)
    # Cố định cho PyTorch (GPU - nếu có)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Đảm bảo các phép toán của cuDNN hoạt động ổn định
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Đã cố định Random Seed: {seed}")
