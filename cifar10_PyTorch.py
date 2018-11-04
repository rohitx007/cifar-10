#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[2]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias = False).cuda()
        self.batchnorm1 = nn.BatchNorm2d(32).cuda()
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias = False).cuda()
        self.batchnorm2 = nn.BatchNorm2d(32).cuda()
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias = False).cuda()
        self.batchnorm3 = nn.BatchNorm2d(64).cuda()
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias = False).cuda()
        self.batchnorm4 = nn.BatchNorm2d(64).cuda()
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1, bias = False).cuda()
        self.batchnorm5 = nn.BatchNorm2d(128).cuda()
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1, bias = False).cuda()
        self.batchnorm6 = nn.BatchNorm2d(128).cuda()
        self.pool = nn.MaxPool2d(2, 2).cuda()
        self.dropout1 = nn.Dropout(0.2).cuda()
        self.dropout2 = nn.Dropout(0.3).cuda()
        self.dropout3 = nn.Dropout(0.4).cuda()
        self.fc1 = nn.Linear(2048, 120).cuda()
        self.fc2 = nn.Linear(120, 84).cuda()
        self.fc3 = nn.Linear(84, 10).cuda()

    def forward(self, x):
        x = x.to(device)
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.dropout1(self.pool(x))
        x = self.batchnorm3(F.relu(self.conv3(x)))
        x = self.batchnorm4(F.relu(self.conv4(x)))
        x = self.dropout2(self.pool(x))
        x = self.batchnorm5(F.relu(self.conv5(x)))
        x = self.batchnorm6(F.relu(self.conv6(x)))
        x = self.dropout3(self.pool(x))
        x = x.view(4, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)


# In[3]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)


# In[ ]:


for epoch in range(2):  # loop over the dataset multiple times
    print('Hello')
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted = predicted.to(device)
            labels = labels.to(device)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

print('Finished Training')


# In[ ]:


# Confusion matrix result
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd

df_cm = pd.DataFrame(cm, range(10),range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()


# In[ ]:




