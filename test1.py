import torch.nn as nn
import torch

linear = nn.Linear(3, 4)
print(hasattr(linear, 'weight'))
