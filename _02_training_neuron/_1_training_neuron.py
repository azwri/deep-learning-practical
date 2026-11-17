#%%
import torch
from torch import nn

X1 = torch.tensor([[10.0]], dtype=torch.float32) # input: temperature in C
y1 = torch.tensor([[50.0]], dtype=torch.float32) # actul value: temperaturein F

X2 = torch.tensor([[37.78]], dtype=torch.float32) # input: temperature in C
y2 = torch.tensor([[100.0]], dtype=torch.float32) # actul value: temperaturein F

model = nn.Linear(1, 1)
# for i in model.parameters():
#     print(i)
loss_fn = torch.nn.MSELoss()
# print(loss_fn(
#     torch.tensor([[37.78]], dtype=torch.float32), # input 
#     torch.tensor([[100.0]], dtype=torch.float32) # actul value
# ))
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)

print("Model before trainin:", model.weight)
# training pass
for i in range(0, 100000):
    optimizer.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()
    
    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()
    
    if i % 1000 == 0:
        print("bais =", model.bias)
        print("weight =", model.weight)
    
print("Model after trainin:", model.weight)


y1_pred = model(X1)
y2_pred = model(X2)

print("y1_pred =", y1_pred, "y1_actul =", y1)
print("y2_pred =", y2_pred, "y2_actul =", y2)
# %%
