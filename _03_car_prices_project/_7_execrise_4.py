#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
# %%

X_mean = torch.load('./model/execrise2/X_mean.pt')
X_std = torch.load('./model/execrise2/X_std.pt')
y_mean = torch.load('./model/execrise2/y_mean.pt')
y_std = torch.load('./model/execrise2/y_std.pt')

model = nn.Linear(3, 1)
model.load_state_dict(
    state_dict=torch.load('./model/execrise2/model.pt', weights_only=True)
)
# %%
model.eval()

X_data = torch.tensor([
    [5, 10000, 1],
], dtype=torch.float32)

with torch.no_grad():
    X_data_normalize = (X_data - X_mean) / X_std
    y_pred = model(X_data_normalize)
    y_pred = (y_pred * y_std + y_mean)
    print(y_pred)
# %%
