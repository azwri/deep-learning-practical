#%%
import torch

b = 32
w1 = 1.8

X1 = torch.tensor(10)

y_pred = 1 * b + X1 * w1
print(y_pred)

X = torch.tensor([10, 38, 100, 150])

y_pred = 1 * b + X * w1
print(y_pred)
# %%
