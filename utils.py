import time
import torch
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 耗时: {end - start:.2f} 秒")
        return result
    return wrapper

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)