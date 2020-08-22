import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return self.m * x


def test_model():
    m = Model()
    x = torch.randn(3, 100, 100)
    assert torch.all(m(x) == x)
    print("All checks passed")