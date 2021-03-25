#Bidirectional Recurrent Neural Network

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
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device).squeeze(1)
            targets = targets.to(device)
            
            output = model(data)
            _,preds = output.max(1)
            num_correct += (preds == targets).sum()
            num_samples += preds.size(0)
            
        acc = float(num_correct) / float(num_samples)
    return acc

#hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5

#create Bidirectional RNN
class BLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, 
                            num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        #multilple the num_layers by 2 since we have inputs from both side
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out
    

#load data
train_ds = datasets.MNIST(root='input/', train=True, 
                          transform=transforms.ToTensor(), download=True)
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_ds = datasets.MNIST(root='input/', train=False, 
                          transform=transforms.ToTensor(), download=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)


#initialize the network
model = BLSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)

#loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
train_loss = []
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dl):
        data = data.to(device).squeeze(1) #data: batch_size x 28 x 28
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
        