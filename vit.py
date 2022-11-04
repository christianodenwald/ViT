import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) 

class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x

class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x

class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x

class ViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x

def test(x):
    print("this is a test:", x)