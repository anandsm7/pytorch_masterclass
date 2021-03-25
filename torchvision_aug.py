import torch
import torchvision.transforms as tfms 
from torchvision.utils import save_image
from image_dataset import CatsAndDogsDataset
import matplotlib.pyplot as plt

transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((256, 256)),
    tfms.RandomCrop((224, 224)),
    tfms.ColorJitter(),
    tfms.RandomRotation(degrees=45),
    tfms.RandomGrayscale(p=0.2),
    tfms.RandomHorizontalFlip(p=0.5), #flip image horizontal
    tfms.RandomVerticalFlip(p=0.05),
    tfms.ToTensor(),
    tfms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),# (pixel - mean) / std
                                     #for each channel we need to 
                                     #find mean and std across all pixel values
    ]
)
dataset = CatsAndDogsDataset(csv_path='./input/image_aug/cats_dogs.csv',
                             root_dir= './input/image_aug',
                             transform=transforms
                             )

for i, (img, label) in enumerate(dataset):
    print(img.shape)
    save_image(img, f"./input/image_aug/my_aug/img_{i}.png")

plt.imshow(img.permute([1,2,0]))
plt.show()