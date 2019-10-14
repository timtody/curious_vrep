import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super().__init()
        self.model = nn.Sequential(nn.Conv2d(1, 20, 5),
                                  nn.ReLU(),
                                  nn.Conv2d(20, 64, 5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(64, 64, 3),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(128),
                                  nn.Linear(128),
                                  nn.Linear(n_discrete_actions),
                                  nn.Softmax())

        self.optim = optim.adadelta()
        self.loss = nn.CrossEntropyLoss()

    def fit(self, inputs, targets):
        out = self.model(inputs)
        loss = self.loss(out, targets)
        self.optim.zero_grads()
        self.optim.backward()

        return loss
