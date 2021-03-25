#Simple Convolution Neural network

#imports
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torch.utils.data import DataLoader,random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from image_dataset import CatsAndDogsDataset

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

    
#hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 1
csv_file = './input/cat_vs_dogs/cats_dogs.csv'
root_path = './input/cat_vs_dogs/train/'

#load data

dataset = CatsAndDogsDataset(csv_path=csv_file, 
                             root_dir=root_path, 
                             transform=transforms.ToTensor())
train_set, valid_set = random_split(dataset, [20000, 5000])
train_dl = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
valid_dl = DataLoader(dataset=valid_set,batch_size=batch_size,shuffle=True)


#initialize the network
model = torchvision.models.googlenet(pretrained=True)
model.fc = nn.Linear(in_features=1024,out_features= 2)
model.to(device);

#loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
train_loss = []
model.train()
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

test_acc = accuracy(model, valid_dl)
print(f"test accuracy: {test_acc}")
       

        
        