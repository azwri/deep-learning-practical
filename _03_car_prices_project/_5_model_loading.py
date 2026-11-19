#%%
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pylab as plt


X_mean = torch.load('./model/X_mean.pt')
X_std = torch.load('./model/X_std.pt')
y_mean = torch.load('./model/y_mean.pt')
y_std = torch.load('./model/y_mean.pt')

model = nn.Linear(2, 1)
model.load_state_dict(
    torch.load('./model/model.pt', weights_only=True)
)

model.eval()
X_data = torch.tensor([
    [5, 22770],
    [2, 100000],
    ], dtype=torch.float32) # price = 184606


with torch.no_grad():
    test_pred = model((X_data - X_mean) / X_std)
    print(test_pred * y_std + y_mean)

# %%
