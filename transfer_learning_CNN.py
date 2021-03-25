## Transfer learning using VGG
#imports
import sys
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datatsets
import torchvision.transforms as tfms
import torchvision
import matplotlib.pyplot as plt 

#check accuracy on traininig & test to see how good the model 
def accuracy(model, data_loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            output = model(data)
            _,preds = output.max(1)
            num_correct += (preds == targets).sum()
            num_samples += preds.size(0)
            
        acc = float(num_correct) / float(num_samples)
    return acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

#loading the pretrained model
#dummy layer which does nothing
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

model = torchvision.models.vgg16(pretrained=True)

# If you want to do finetuning then set requires_grad = False
# Remove these two lines if you want to train entire model,
# and only want to load the pretrain weights.

for param in model.parameters():
    param.requires_grad=False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100, num_classes))  
model.to(device=device);

#Load data
train_ds = datatsets.CIFAR10(root='./input', train=True,
                                  transform=tfms.ToTensor(),download=True)
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)

test_ds = datatsets.CIFAR10(root='./input', train=True,
                                  transform=tfms.ToTensor(),download=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
train_loss = []
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dl):
        data = data.to(device) #data: batch_size x 1 x 28 x 28
        targets = targets.to(device)
        
        #forward
        output = model(data)
        loss = criterion(output, targets)
        train_loss.append(loss)
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        
        #gradient update
        optimizer.step()
        
    acc = accuracy(model, train_dl)
    print(f"epoch : {epoch} loss: {loss} acc: {acc}")
 
plt.title("Train Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(train_loss)       
plt.show()

test_acc = accuracy(model, test_dl)
print(f"test accuracy: {test_acc}")
