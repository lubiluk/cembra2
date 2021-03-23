import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import h5py
import cv2
from torch.utils.data import Dataset, DataLoader

PATH = "./data/vision_0.pth"

def imshow(img):
    img = img / 2 + 0.5
    cv2.imshow("camera", img.numpy().transpose((1, 2, 0)))
    cv2.waitKey(1)


class PepperDataset(Dataset):
    def __init__(self, hdf_file, transform=None):
        self.data = h5py.File(hdf_file, "r")
        self.transform = transform

    def __len__(self):
        return len(self.data["camera"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data["camera"][idx]
        position = self.data["object_position"][idx]

        if self.transform:
            image = self.transform(image)

        sample = (image, position)

        return sample


class Transpose(object):
    def __call__(self, sample):
        image, position = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return (image, position)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

trainset = PepperDataset("./data/collect_0_100k.hdf5", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=0)

testset = PepperDataset("./data/collect_0_10k.hdf5", transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(len(trainset))
# show images
imshow(images[0])
# print labels
print(" ".join("%f" % labels[0][j] for j in range(3)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 27 * 37, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 27 * 37)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

net = Net()
net.to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")

torch.save(net, PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(images[0])
print("GroundTruth: ", " ".join("%f" % labels[0][j] for j in range(3)))

net = torch.load(PATH)

predicted = net(images)

print("Predicted: ", " ".join("%f" % predicted[0][j] for j in range(3)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        predicted = net(images)
        total += labels.size(0)
        correct += (np.linalg.norm(predicted - labels, axis=-1) <
                    0.1).sum().item()

print("Accuracy of the network on the 10000 test images: %d %%" %
      (100 * correct / total))
