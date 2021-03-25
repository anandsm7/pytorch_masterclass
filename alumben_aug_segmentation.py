import cv2
import albumentations as A 
import numpy as np 
from utils import plot_examples
from PIL import Image

image = Image.open("./input/image_aug/albu_img/elon.jpeg")
mask = Image.open("./input/image_aug/albu_img/mask.jpeg")
mask2 = Image.open("./input/image_aug/albu_img/second_mask.jpeg")

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
    ]
)

image_lst = [image]
image = np.array(image)
mask = np.array(mask)
mask2 = np.array(mask2)

for i in range(4):
    augmentation = transform(image=image, masks = [mask, mask2])
    augmented_img = augmentation["image"]
    augmented_mask = augmentation["masks"]
    image_lst.append(augmented_img)
    image_lst.append(augmented_mask[0])
    image_lst.append(augmented_mask[1])
    
plot_examples(image_lst)