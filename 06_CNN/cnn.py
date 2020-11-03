# Convolutional Neural Network

import random
import keras
import numpy as np
import matplotlib.pyplot as plt

from helpers import *

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

#classes = ('food', 'not food')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# PATHS
image_path = "/home/leander/uia/ikt450/06_cnn/data/food11_small"  # small version
training_path = os.path.join(image_path, "training")
validation_path = os.path.join(image_path, "validation")
evaluation_path = os.path.join(image_path, "evaluation")


# CONFIG
batch_size = 128
num_classes = 11
epochs = 12
img_rows, img_cols = 128,128
input_shape = (img_rows, img_cols, 1)

# PREPROCESS
x_train, y_train = preprocess(loadImages(training_path))
x_test, y_test = preprocess(loadImages(validation_path))
x_eval, y_eval = preprocess(loadImages(evaluation_path))
x_train = np.array(x_train)
x_test = np.array(x_train)
x_eval = np.array(x_train)


# Keras is picky on the input shape..
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
x_eval = x_eval.reshape(x_eval.shape[0],img_rows,img_cols,1)
# Convert uint8 gray pixels to floats (0-1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_eval = x_eval.astype('float32')
x_train /= 255
x_test /= 255
x_eval /= 255

# Convert labels to categorical one-hot encoding, since this is multi-class
y_train = keras.utils.to_categorical(y_train, num_classes=11)
y_test = keras.utils.to_categorical(y_test, num_classes=11)
y_eval = keras.utils.to_categorical(y_eval, num_classes=11)

# INFO
print('x_train shape:',x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# --------------------------------------------------------------- #

# Show an image
def imshow(img, s=""):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if s and not '\n' in s:
        s = ' '.join(s.plsit())
        p = s.find(' ', int(len(s)/2))
        s = s[:p]+"\n"+s[p+1:]
    plt.text(0, -20, s)
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
s = ' '.join('%5s' % classes[labels[j]] for j in range(16))
print(s)
imshow(torchvision.utils.make_grid(images), s)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backwords()
        optimizer.step()
        running_loss += loss.item()
        if(i % 100 == 0):
            print(epoch, i, running_loss/(i+1))

dataiter = iter(testloader)
images, labels = dataiter.next()

outputs = net(images)
_, predicted = torch.max(outputs, 1)
s1 = "Pred:"+' '.join('%5s' % classes[predicted[j]] for j in range(16))
s2 = "Actual:"+' '.join('%5s' % classes[labels[j]] for j in range(16))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(16)))
# print images
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
imshow(torchvision.utils.make_grid(images),s1+"\n"+s2)
