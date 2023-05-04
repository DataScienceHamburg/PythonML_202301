#%% packages
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
os.getcwd()

#%% transform, load data
transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.Grayscale(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5,))
    ]
)

BATCH_SIZE = 4
train_image_path = 'data/train'
test_image_path = 'data/test'
trainset = torchvision.datasets.ImageFolder(root=train_image_path, transform=transform)
testset = torchvision.datasets.ImageFolder(root=test_image_path, transform=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# %% visualize images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images, nrow=2))
# %% Neural Network setup
class ImageClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

input = torch.rand((1, 1, 32, 32))
model = ImageClassificationNet()
model(input).shape

#%% init model
model = ImageClassificationNet()      
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# %% training
NUM_EPOCHS = 30
train_losses = []
for epoch in range(NUM_EPOCHS):
    loss_current_epoch = 0
    for i, data in enumerate(trainloader, 0):
        X_batch, y_batch = data
        
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(X_batch)
                
        # calc losses
        loss = loss_fn(y_pred.squeeze(), y_batch.float())
        loss_current_epoch += loss.item()
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch {epoch}/{NUM_EPOCHS}, Step {i+1}/{len(trainloader)}')
    train_losses.append(loss_current_epoch)

#%% train losses graph
import seaborn as sns
sns.lineplot(x=list(range(len(train_losses))), y= train_losses)

# %% test
y_test = []
y_test_pred = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_pred.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {acc*100:.2f} %')
# %%
# We know that data is balanced, so baseline classifier has accuracy of 50 %.