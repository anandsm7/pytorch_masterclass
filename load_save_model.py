#Simple Convolution Neural network

#imports
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

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

#CNN 
class CNNet(nn.Module):
    def __init__(self,in_channels, num_classes):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8,
                            kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,
                               kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        out = self.fc1(x)
        
        return out
    
#to save checkpoint
def save_checkpoint(state, filename="./input/checkpoint.pth.tar"):
    print("save checkpoint")
    torch.save(state, filename)
    
#load model
def load_checkpoint(checkpoint):
    print("loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    
#hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5
load_model = True
MODEL_PATH = './input/checkpoint.pth.tar'

#load data
train_ds = datasets.MNIST(root='input/', train=True, 
                          transform=transforms.ToTensor(), download=True)
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_ds = datasets.MNIST(root='input/', train=False, 
                          transform=transforms.ToTensor(), download=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)


#initialize the network
model = CNNet(in_channels=in_channels, num_classes=num_classes).to(device)

#loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load(MODEL_PATH))

#train network
train_loss = []
model.train()
for epoch in range(num_epochs):
    checkpoint = {'state_dict': model.state_dict(), 
              'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)
    
    
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
       

        
        