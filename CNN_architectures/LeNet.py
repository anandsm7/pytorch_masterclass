import torch
import torch.nn as nn 

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, 
                               kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=(5,5), stride=(1, 1), padding=(0, 0 ))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=(5,5), stride=(1, ), padding=(0, 0))
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        print("input -> ",x.shape)
        x = self.tanh(self.conv1(x))
        print("conv1 -> ",x.shape)
        x = self.pool(x)
        print("pool1 -> ",x.shape)
        x = self.tanh(self.conv2(x))
        print("conv2 -> ",x.shape)
        x = self.pool(x)
        print("pool2 -> ",x.shape)
        x = self.tanh(self.conv3(x)) # from 120x1x1 to 120
        print("conv3 -> ",x.shape)
        x = x.reshape(x.shape[0], -1)
        print("conv3 reshape -> ",x.shape)
        x = self.relu(self.fc1(x))
        print("fc1 -> ",x.shape)
        out = self.fc2(x)
        print("fc2 -> ",out.shape)
        
        return out
    
if __name__ == "__main__":
    x = torch.rand((32, 1, 32, 32))
    model = LeNet()
    print(model)
    out = model(x)