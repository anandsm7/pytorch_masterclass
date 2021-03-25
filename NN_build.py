#simple full connected Neural network
#imports
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
            data = data.reshape(data.shape[0], -1)
            
            output = model(data)
            _,preds = output.max(1)
            num_correct += (preds == targets).sum()
            num_samples += preds.size(0)
            
        acc = float(num_correct) / float(num_samples)
    return acc

#create NN
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


#hyperparameters
input_size = 784
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 10

#load data
train_ds = datasets.MNIST(root='input/', train=True, 
                          transform=transforms.ToTensor(), download=True)
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_ds = datasets.MNIST(root='input/', train=False, 
                          transform=transforms.ToTensor(), download=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)


#initialize the network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dl):
        data = data.to(device) #data: batch_size x 1 x 28 x 28
        targets = targets.to(device)
        
        data = data.reshape(data.shape[0],-1) # to shape bs x 784
        
        #forward
        output = model(data)
        loss = criterion(output, targets)
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        
        #gradient update
        optimizer.step()
        
    acc = accuracy(model, train_dl)
    print(f"epoch : {epoch} loss: {loss} acc: {acc}")
        

test_acc = accuracy(model, test_dl)
print(f"test accuracy: {test_acc}")