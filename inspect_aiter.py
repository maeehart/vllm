import torch
import aiter
try:
    print(dir(torch.ops.aiter))
except Exception as e:
    print(e)
