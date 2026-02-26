import torch

positions = torch.randn(10, 3)
x = positions.expand(10, -1, -1)
print(x.shape)
                         