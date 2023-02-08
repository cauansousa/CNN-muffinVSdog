import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.ImageFolder('C:\\Users\\Cauan\\OneDrive - FEI\\Dataset\\train\\', transform=transform)
test_dataset = torchvision.datasets.ImageFolder('C:\\Users\\Cauan\\OneDrive - FEI\\Dataset\\test\\', transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
dataiter = iter(train_loader)
images, labels = next(dataiter)

# create convolutional neural network
conv1 = nn.Conv2d(3, 32, 5)
pool1 = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(32, 64, 5)
pool2 = nn.MaxPool2d(2, 2)

print("Cru:",images.shape)
x = conv1(images)
print("Conv1:",x.shape)
x = pool1(x)
print("Pool1:",x.shape)
x = conv2(x)
print("Conv2:",x.shape)
x = pool2(x)
print("Pool2:",x.shape)


