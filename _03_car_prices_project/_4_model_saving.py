#%%
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pylab as plt
import os

df = pd.read_csv('./data/used_cars_clean.csv')

X = torch.column_stack([
    torch.tensor(df['age'], dtype=torch.float32),
    torch.tensor(df['milage'], dtype=torch.float32)
])
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X- X_mean) / X_std

y = torch.tensor(
    df['price'], dtype=torch.float32
).reshape((-1, 1))

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) /y_std

model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

losses = []
for i in range(0, 500):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

plt.plot(losses)
plt.show()
# %%
prediction = model(X)
print(prediction)

# %%
X_data = torch.tensor([
    [5, 22770],
    [2, 100000],
    ], dtype=torch.float32) # price = 184606
test_pred = model((X_data - X_mean) / X_std)
print(test_pred * y_std + y_mean)

# %%
# save model
if not os.path.isdir('./model'):
    os.mkdir('./model')

torch.save(X_mean, './model/X_mean.pt')
torch.save(X_std, './model/X_std.pt')
torch.save(y_mean, './model/y_mean.pt')
torch.save(y_std, './model/y_std.pt')

torch.save(model.state_dict(), './model/model.pt')
# %%
