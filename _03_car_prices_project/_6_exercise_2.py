#%%
import pandas as pd
import numpy as np
import torch
from torch import nn
import os

#%%
X_mean = torch.load('./model/exerice/X_mean.pt')
X_std = torch.load('./model/exerice/X_std.pt')
y_mean = torch.load('./model/exerice/y_mean.pt')
y_std = torch.load('./model/exerice/y_std.pt')

model = nn.Linear(3, 1)
model.load_state_dict(
    state_dict=torch.load('./model/exerice/model.pt', weights_only=True)
)

model.eval()

X_data = torch.tensor([
    [5, 10000, 0],
    [2, 10000, 0],
    [5, 20000, 0],
], dtype=torch.float32)
#%%
with torch.no_grad():
    X_data = model((X_data - X_mean) / X_std)
    y_pred = X_data * y_std + y_mean
    print(y_pred)
    
# %%
