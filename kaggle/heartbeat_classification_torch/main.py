# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 00:15:39 2020

@author: hanwo
"""

import math
import random
import pickle
import itertools
import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 

from sklearn.utils import shuffle

from scipy.signal import resample

import matplotlib.pyplot as plt

np.random.seed(42)

import pickle
from sklearn.preprocessing import OneHotEncoder

import os
print(os.listdir("../heartbeat"))

df = pd.read_csv("../heartbeat/mitbih_train.csv", header=None)
df2 = pd.read_csv("../heartbeat/mitbih_test.csv", header=None)
df3 = pd.concat([df, df2], axis=0)

M = df3.values
X = M[:, :-1]
y = M[:, -1].astype(int)


C0 = np.argwhere(y == 0).flatten()
C1 = np.argwhere(y == 1).flatten()
C2 = np.argwhere(y == 2).flatten()
C3 = np.argwhere(y == 3).flatten()
C4 = np.argwhere(y == 4).flatten()

x = np.arange(0, 187)*8/1000

# plt.figure(figsize=(20,12))
# plt.plot(x, X[C0, :][0], label="Cat. N")
# plt.plot(x, X[C1, :][0], label="Cat. S")
# plt.plot(x, X[C2, :][0], label="Cat. V")
# plt.plot(x, X[C3, :][0], label="Cat. F")
# plt.plot(x, X[C4, :][0], label="Cat. Q")
# plt.legend()
# plt.title("1-beat ECG for every category", fontsize=10)
# plt.ylabel("Amplitude", fontsize=8)
# plt.xlabel("Time (ms)", fontsize=8)
# plt.show()


def stretch(x):
    l = int(187 * (1 + (random.random()-0.5)/3))
    y = resample(x, l)
    if l < 187:
        y_ = np.zeros(shape=(187, ))
        y_[:l] = y
    else:
        y_ = y[:187]
    return y_

def amplify(x):
    alpha = (random.random()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor

def augment(x):
    result = np.zeros(shape= (4, 187))
    for i in range(3):
        if random.random() < 0.33:
            new_y = stretch(x)
        elif random.random() < 0.66:
            new_y = amplify(x)
        else:
            new_y = stretch(x)
            new_y = amplify(new_y)
        result[i, :] = new_y
    return result

result = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)
classe = np.ones(shape=(result.shape[0],), dtype=int)*3
X = np.vstack([X, result])
y = np.hstack([y, classe])

subC0 = np.random.choice(C0, 800)
subC1 = np.random.choice(C1, 800)
subC2 = np.random.choice(C2, 800)
subC3 = np.random.choice(C3, 800)
subC4 = np.random.choice(C4, 800)

X_test = np.vstack([X[subC0], X[subC1], X[subC2], X[subC3], X[subC4]])
y_test = np.hstack([y[subC0], y[subC1], y[subC2], y[subC3], y[subC4]])

X_train = np.delete(X, [subC0, subC1, subC2, subC3, subC4], axis=0)
y_train = np.delete(y, [subC0, subC1, subC2, subC3, subC4], axis=0)

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=0)

X_train = np.expand_dims(X_train, 2)
X_test = np.expand_dims(X_test, 2)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

num_classes=5
num_epochs = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


X_train = np.array(X_train)
y_train = np.array(y_train)
# y_train = np.expand_dims(y_train, 1)

X_test = np.array(X_test)
y_test = np.array(y_test)
# y_test = np.expand_dims(y_test, 1)


X_train = torch.from_numpy(X_train).float().permute(0,2,1)
y_train = torch.from_numpy(y_train).long()

X_test = torch.from_numpy(X_test).float().permute(0,2,1)
y_test = torch.from_numpy(y_test).long()


train = TensorDataset(X_train, y_train)

train_loader = DataLoader(train, batch_size=32, shuffle=True)

test = TensorDataset(X_test, y_test)
test_loader = DataLoader(test, batch_size=32, shuffle=True)


# feature, depth, n_obs = X_train.shape

class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16,32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
            )
        self.layer3 = nn.Sequential(
            nn.Linear(1440, 500),
            nn.Softmax(),
            nn.Linear(500, num_classes)
            )
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)
        return out
    
model = Net(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (X_train, y_train) in enumerate(train_loader):
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for X_test, y_test in test_loader:
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
