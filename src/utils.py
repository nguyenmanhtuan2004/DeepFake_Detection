import os, torch, random, numpy as np
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True




def get_device(prefer_gpu: bool = True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')




def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)