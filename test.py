import torch.nn as nn
import torch

x = torch.randn(2, 3, 5, 5)
avg_pool = nn.AdaptiveAvgPool2d(1)
lr = nn.Linear(3, 2, bias=False)

print(x[0, 0].mean())
y = avg_pool(x)
print(y[0, 0])
y = lr(y.view(2, 3))

print(y.size())