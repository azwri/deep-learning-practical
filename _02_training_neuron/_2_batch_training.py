#%%
import torch
from torch import nn

X = torch.tensor([[10.0], [37.78]], dtype=torch.float32) # input: temperature in C
y = torch.tensor([[50.0], [100.00]], dtype=torch.float32) # actul value: temperaturein F

x_mean = X.mean()
x_std = X.std()
x_norm = (X - x_mean) / x_std

model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

for i in range(0, 10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    
    if i % 1000 == 0:
        print("bais =", model.bias)
        print("weight =", model.weight)
    

w = model.weight.item()
b = model.weight.item()
w_real = w / x_std.item()
b_real = b - w * x_mean.item() / x_std.item()
print("weight =", w_real, " bias =", b_real)

y_pred = model(X)
print("y_pred =", y_pred)
print( "y1_actul =", y)

#%%
measurement = torch.tensor([[37.5]], dtype=torch.float32)
model.eval()
with torch.no_grad():
    prediction = model((measurement - x_mean) / x_std)
    print(prediction)