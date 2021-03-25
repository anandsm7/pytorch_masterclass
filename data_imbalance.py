import os
import torch
import torch.nn as nn 
import torchvision.transforms as tfms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset, WeightedRandomSampler

#method for dealing with imbalanced dataset
# 1.Oversampling
# 2.class weighting

def get_loader(root_dir, batch_size):
    my_transform = tfms.Compose(
        [
            tfms.Resize((224, 224)),
            tfms.ToTensor(),
        ]
    )
    
    ds = ImageFolder(root=root_dir, transform=my_transform)
    # class_weights = [1, 50]
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1*len(files))
    print(f"class wieghtage: {class_weights}")

    sample_weights = [0] * len(ds)
    
    for idx, (data, label) in enumerate(ds):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
        
    sampler = WeightedRandomSampler(sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    
    return loader

def main():
    loader = get_loader(root_dir='./input/cat_dogs/', batch_size=8)
    num_cats = 0
    num_dogs = 0
    for epoch in range(10):
        for x, y in loader:
            num_cats += torch.sum(y==0)
            num_dogs += torch.sum(y==1)
    print(num_cats, num_dogs)
if __name__ == "__main__":
    main()
