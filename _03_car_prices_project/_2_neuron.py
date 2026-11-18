#%%
import pandas as pd
import numpy as np
import torch
from torch import nn

#%%
df = pd.read_csv('./data/used_cars_clean.csv')
print(df.info())

#%%
# inputs(age, milage) output(price)
milage = df['milage']
age = df['age']
price = df['price']

# convert data to tensors
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])

y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

#%%
model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000000001)
# %%
for i in range(0, 1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    print(loss)
    
    # if i % 100 == 0:
    #     print(model.bias)
    #     print(model.weight)


# %%
prediction = model(X)
print(prediction)

# %%
print(
    model(torch.tensor([
        [5, 10000]
    ], dtype=torch.float32))
)
# %%
