import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sim_data_prep as data
import params as p
import imp
imp.reload(data)
imp.reload(p)

params = params.Params()
dataset = data.SimDataset(params)
loader = DataLoader(dataset, batch_size=5)

K = 1
d = 2048
T =30
h = 32
w = 64
b = 10
epochs = 100
# data_root = '../../data/simulated/sim_dataset/sim000'
# Agt = torch.t(torch.from_numpy(np.load(os.path.join(data_root, 'A.npy'))))
# Cgt = torch.from_numpy(np.load(os.path.join(data_root, 'C.npy')))
# Xgt = (torch.mm(Agt, Cgt)).float()
# print(Xgt.size())

batch = next(iter(loader))
print(batch.size())

A = torch.randn((d, K), requires_grad=True)
C = torch.randn((K, T), requires_grad=True)

optimizer = torch.optim.SGD(params=[A, C], lr=1)
criterion = torch.nn.MSELoss()

losses = []
Agrads = []
Cgrads = []
for i in range(epochs):
    X = torch.matmul(A, C)
    if i==0:
        print(X.size())
    loss = criterion(X, Xgt)
    optimizer.zero_grad()
    loss.backward()
    Agrads.append(A.grad.clone().mean())
    Cgrads.append(C.grad.clone().mean())
    optimizer.step()
    losses.append(loss.detach())

plt.plot(np.arange(len(losses)), losses)
plt.title('losses')
plt.show()
plt.close()

A = np.reshape(A.detach().numpy(), (h, w))
plt.imshow(A)
plt.show()

Agt = np.reshape(Agt.numpy(), (h, w))
plt.imshow(Agt)
plt.show()

plt.plot(np.arange(C.size(1)), C.detach().numpy()[0])
plt.plot(np.arange(Cgt.size(1)), Cgt.numpy()[0])
plt.show()
