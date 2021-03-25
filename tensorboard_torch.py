import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms as tfms
from torchvision.utils import make_grid
from torch.utils.data import (
    DataLoader, Dataset
)
from torch.utils.tensorboard import SummaryWriter


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, 
            kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return x
    
#set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyperparameters
in_channels = 1 
num_classes = 10
num_epochs = 5 

#Load data 
train_dataset = datasets.MNIST(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True
)

#To do hyperparameter search
batch_sizes = [128,256,1024]
learning_rates = [0.01,0.001,0.0001]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        #init network
        model = CNN(in_channels=in_channels, num_classes=num_classes)
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), 
                               lr=learning_rate, 
                               weight_decay=0.0)
        #initialize the tensorboard summary writer
        writer = SummaryWriter(
            f"runs/MNIST/BS {batch_size} LR {learning_rate}"
        )
        #visualize model in tensorboard
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(device))
        writer.close()
        
        for epoch in range(num_epochs):
            losses = []
            accs = []
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #calulate training accuracy
                features = data.reshape(data.shape[0], -1)
                img_grid = make_grid(data)
                _, predictions = outputs.max(1)
                num_correct = (predictions == targets).sum()
                train_acc = float(num_correct) / float(data.shape[0])
                accs.append(train_acc)
                
                #plot to tensorboard
                class_labels = [classes[label] for label in predictions]
                writer.add_image("mnist_images", img_grid)
                writer.add_histogram("fc", model.fc1.weight)
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar("Training acc", train_acc, global_step=step)
                
                if batch_idx == 230:
                    writer.add_embedding(
                        features,
                        metadata=class_labels,
                        label_img=data,
                        global_step=batch_idx
                    )
                step += 1
                
            writer.add_hparams(
                {"lr": learning_rate, "batchsize": batch_size},
                {
                    "accuracy": sum(accs) / len(accs),
                    "loss": sum(losses) / len(losses),
                },
            )
    
    