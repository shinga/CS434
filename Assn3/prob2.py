# @Author: Arthur Shing
# @Date:   2018-05-13T18:57:58-07:00
# @Filename: prob.py
# @Last modified by:   Arthur Shing
# @Last modified time: 2018-05-14T12:37:48-07:00
# Much code grabbed from Markus Koskela,
# https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.fc1 = nn.Linear(32*32*3, 100)
# 		self.fc1_drop = nn.Dropout(0.2)
# 		self.fc2 = nn.Linear(100, 10)
#
# 	def forward(self, x):
# 		x = x.view(-1, 32*32*3) #-1 means don't know how many rows to reshape to
# 		x = F.relu(self.fc1(x))
# 		x = self.fc1_drop(x)
# 		return F.log_softmax(self.fc2(x))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(50, 50)
        # self.fc2_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        # x = F.logsigmoid(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc2_drop(x)
        # return F.log_softmax(self.fc3(x))
        return F.log_softmax(self.fc2(x))

model = Net()
if cuda:
    model.cuda()

# LEARNRATE = 0.01
# optimizer = optim.SGD(model.parameters(), lr=LEARNRATE, momentum=0.5)

print(model)

batch_size = 10000


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                   ])),
    batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                       # transforms.Normalize((128,), (128,))
                   ])),
    batch_size=batch_size, shuffle=False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(32,32,3))
    C = X_train[i,:,:,:].numpy().reshape(32,32,3)
    plt.title('Class: '+str(y_train[i]))

plt.imsave("mnist.png",X_train[i,:,:,:].numpy().reshape(32,32,3))



def main():
    LEARNRATE = float(sys.argv[1])
    optimizer = optim.SGD(model.parameters(), lr=LEARNRATE, momentum=0.5)
    epochs = 10

    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch, optimizer)
        validate(lossv, accv, optimizer)

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), lossv)
    plt.title('validation loss')
    plt.savefig(str(LEARNRATE) + "RELUloss.png")

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), accv)
    plt.title('validation accuracy');
    plt.savefig(str(LEARNRATE) + "RELUacc.png")



def train(epoch, optimizer, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))

def validate(loss_vector, accuracy_vector, optimizer):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))



if __name__ == "__main__":
    main()
