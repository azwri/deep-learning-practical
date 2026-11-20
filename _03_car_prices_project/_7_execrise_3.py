#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os


df = pd.read_csv('./data/no_accident_used_cars_clean.csv')
# %%

X = torch.column_stack([
    torch.tensor(df['age'], dtype=torch.float32),
    torch.tensor(df['milage'], dtype=torch.float32),
    torch.tensor(df['accident_col'], dtype=torch.float32)
])

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
print(X)
# %%
y = torch.tensor(
    df['price'],
    dtype=torch.float32).reshape((-1, 1))

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std
print(y)
# %%

model = nn.Linear(3, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
losses = []

for i in range(0, 5000):
    model.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

plt.plot(losses)
plt.show()
# %%
os.makedirs('./model/execrise2', exist_ok=True)
torch.save(X_mean, './model/execrise2/X_mean.pt')
torch.save(X_std, './model/execrise2/X_std.pt')
torch.save(y_mean, './model/execrise2/y_mean.pt')
torch.save(y_std, './model/execrise2/y_std.pt')
torch.save(model.state_dict(), './model/execrise2/model.pt')
# %%
