import torchvision
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

data_path ="/home/sam/Dataset/modify/root7/"

transform_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

image_data = torchvision.datasets.ImageFolder(
  root=data_path, transform=transform_img
)
image_data_loader = DataLoader(
  image_data,

  # batch size is whole datset
  batch_size=len(image_data), 
  shuffle=False, 
  num_workers=0)

def mean_std(loader):
  images, lebels = next(iter(loader))
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)

""" Result
mean and std:
 tensor([0.4023, 0.4194, 0.3777]) tensor([0.2037, 0.0968, 0.1329])
"""

image_path = "/home/sam/Dataset/modify/root7/touch/_60/left_11_1.png"
img = cv2.imread(image_path)
frame = Image.fromarray(img)
cv2.imshow('image', img)

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.4023, 0.4194, 0.3777], [0.2037, 0.0968, 0.1329])
    transforms.Normalize([0,0,0], [1,1,1])
])

img_trans = transform(img)

arr = img_trans.numpy()
max_value = arr.max()
arr = arr * 255 / max_value
mat = np.uint8(arr)
mat = mat.transpose(1, 2, 0)
cv2.imshow('mat', mat)

cv2.waitKey(0)