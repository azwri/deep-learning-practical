#%%
import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('./data/used_cars_clean.csv')
print(df['accident'].value_counts())
print(df['accident'].isna().sum())

accident_col = df['accident'] == 'None reported'
accident_col = accident_col.astype(int)

df['accident_col'] = accident_col
df.to_csv('./data/no_accident_used_cars_clean.csv', index=False)
# %%

X = torch.column_stack([
    torch.tensor(df['age'], dtype=torch.float32),
    torch.tensor(df['milage'], dtype=torch.float32),
    torch.tensor(df['accident_col'], dtype=torch.float32),
])
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
print(X)

y = torch.tensor(
    df['price'], dtype=torch.float32
).reshape((-1, 1))
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

print(y)
# %%

model = nn.Linear(3, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
losses = []

for i in range(0, 10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

plt.plot(losses)
plt.show()

#%%

os.makedirs('./model/exerice', exist_ok=True)

torch.save(X_mean, './model/exerice/X_mean.pt')
torch.save(X_std, './model/exerice/X_std.pt')
torch.save(y_mean, './model/exerice/y_mean.pt')
torch.save(y_std, './model/exerice/y_std.pt')
torch.save(model.state_dict(), './model/exerice/model.pt')

# %%
