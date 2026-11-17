#%%
import torch
from torch import nn

X = torch.tensor([[10.0], [37.78]], dtype=torch.float32) # input: temperature in C
y = torch.tensor([[50.0], [100.00]], dtype=torch.float32) # actul value: temperaturein F

model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)

for i in range(0, 100000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    
    if i % 1000 == 0:
        print("bais =", model.bias)
        print("weight =", model.weight)
    


y_pred = model(X)
print("y_pred =", y_pred, "y1_actul =", y)

#%%
measurement = torch.tensor([[37.5]], dtype=torch.float32)
model.eval()
with torch.no_grad():
    prediction = model(measurement)
    print(prediction)