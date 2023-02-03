import torch
import cv2
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 61 * 61, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = './foto/'

image = 'muffin.jpg'

img = Image.open(path+image)

# Transforme a imagem em um tensor PyTorch
transform = transforms.Compose([
    transforms.ToTensor()
])

img_tensor = transform(img)

img_tensor = img_tensor.unsqueeze(0).to(device)


model = torch.load('muffins_chihuahua.pth')
model.eval()

prd = model(img_tensor)
_, predicted = torch.max(prd.data, 1)

output = "Muffin" if predicted == 1 else "Chihuahua"

print(output)
print(predicted)
