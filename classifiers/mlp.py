import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Net0(nn.Module):
    def __init__(self, n):
        super(Net0, self).__init__()
        self.fc1 = nn.Linear(n, 20)
        self.fc2 = nn.Linear(20, 50)
        self.fc3 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))
        return x

    def predict_proba(self, inp):
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.float32)
        else:
            inp = inp.float()
        out = self.forward(inp).detach()
        out = torch.cat([1 - out, out], axis=-1)
        return out.numpy()

    def predict(self, inp):
        out = self.predict_proba(inp)
        y_pred = np.argmax(out, axis=-1)
        return y_pred


def train(model, X, y, lr=1e-3, num_epoch=100, verbose=False):
    model.train()
    criterion = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    loss_diff = 1.0
    prev_loss = 0.0
    num_stable_iter = 0
    max_stable_iter = 3

    for i in range(num_epoch):
        y_pred = model(X)
        loss = criterion(y_pred.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print("Iter %d: loss: %f" % (i, loss.data.item()))

        loss_diff = prev_loss - loss.data.item()

        if loss_diff <= 1e-7:
            num_stable_iter += 1
            if num_stable_iter >= max_stable_iter:
                break
        else:
            num_stable_iter = 0

        prev_loss = loss.data.item()

    model.eval()
