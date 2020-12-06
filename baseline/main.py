import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import random
import matplotlib.pyplot as plt
random_seed = 777
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

df = pd.read_csv("Dataset_spine.csv")

df['Class_att'] = df['Class_att'].astype('category')
encode_map = {
    'Abnormal': 1,
    'Normal': 0
}

df['Class_att'].replace(encode_map, inplace=True)

X = df.iloc[:, 0:-1]
y = df.iloc[:, -2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.001


## train data
class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train),
                       torch.FloatTensor(y_train))


## test data
class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = testData(torch.FloatTensor(X_test))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()

        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BinaryClassifier()
summary(model, (1, 13),verbose=2)

model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


model.train()
old_loss = 100000
accs = []
losses = []
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch.unsqueeze(1))
        y_batch = y_batch.unsqueeze(1)

        loss = criterion(y_pred, y_batch.unsqueeze(2))
        acc = accuracy_score(torch.round(torch.sigmoid(y_pred.squeeze())).cpu().detach().numpy(), y_batch.squeeze().cpu().detach().numpy())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
    cost = epoch_loss / len(train_loader)
    accuracy = epoch_acc / len(train_loader)
    losses.append(cost)
    accs.append(accuracy)
    print(f'Epoch {e + 0:03}: | Loss: {cost:.5f} | Acc: {accuracy:.3f}')
    if cost < old_loss:
        print("ok")
        old_loss = cost

y_pred_list = torch.tensor([]).to(device)
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch.unsqueeze(1))
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list = torch.cat((y_pred_list, y_pred_tag.squeeze()), dim=0)

# y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

cmt = confusion_matrix(y_test, y_pred_list.cpu())
print(cmt)

plt.plot(losses)
plt.plot(accs)
plt.xlabel("epoch")
plt.show()
