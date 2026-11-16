#%%
import torch
from torch import nn

X = torch.tensor([
    [10.0],
    [30.0],
    [100.0],
    [150.0],
], dtype=torch.float32)

model = nn.Linear(1, 1)
print(model)

y_pred = model(X)
print(y_pred)

model.bias = nn.Parameter(torch.tensor([32.0]))
model.weight = nn.Parameter(torch.tensor([[1.8]])) # matrix [[1,1]]
y_pred = model(X)
print(y_pred)