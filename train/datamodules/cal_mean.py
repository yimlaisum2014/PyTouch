from torch import tensor
import torch
import torchvision
import numpy
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

data_path ="/home/sam/Dataset/modify/root14/"

transform_img = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(256),
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
  print('here')
  images, lebels = next(iter(loader))
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

# mean, std = mean_std(image_data_loader)
# print("mean and std: \n", mean, std)

""" Result
mean and std:
 tensor([0.4023, 0.4194, 0.3777]) tensor([0.2037, 0.0968, 0.1329])
"""
#
# image_path = "/home/sam/Dataset/modify/root7/touch/_60/left_11_1.png"
# img = cv2.imread(image_path)
# frame = Image.fromarray(img)
# cv2.imshow('image', img)
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize([0.4023, 0.4194, 0.3777], [0.2037, 0.0968, 0.1329])
#     transforms.Normalize([0,0,0], [1,1,1])
# ])
#
# img_trans = transform(img)
#
# arr = img_trans.numpy()
# max_value = arr.max()
# arr = arr * 255 / max_value
# mat = np.uint8(arr)
# mat = mat.transpose(1, 2, 0)
# cv2.imshow('mat', mat)
#
# cv2.waitKey(0)

def detect_file(image_path):
        img = cv2.imread(image_path)
        
        from torchvision import transforms
        transform = transforms.Compose([
             transforms.ToTensor(),
        #     # transforms.Normalize([0,0,0], [1,1,1])
        #     # transforms.Normalize([0.0823, 0.0823, 0.0823], [0.1025, 0.1025, 0.1025])
        #     transforms.Normalize([0.3884, 0.4025, 0.3997], [0.1008, 0.0962, 0.1659])
        ])
        
        img_trans = transform(img)
        mean, std = img_trans.mean([1,2]), img_trans.std([1,2])
        # print(mean, std)
        arr = img_trans.numpy()
        max_value = arr.max()
        arr = arr * 255 / max_value
        mat = np.uint8(arr)
        mat = mat.transpose(1, 2, 0)
        #cv2.imshow('mat', mat)
        
        #cv2.waitKey(0)

        return mean, std

import os
import random

image_folder = f"/home/sam/Dataset/modify/root18"
n = 0
id = 0
sum_mean, sum_std = [], []

for root, dirs, names in os.walk(image_folder):
    for name in names:
      n += 1
      if (random.uniform(0,n) < 145): #1 for14
        filename = os.path.join(root, name)
        # print(filename)
        mean, std = detect_file(filename)
        _mean = numpy.array(mean)
        _std = numpy.array(std)
        sum_mean.append(_mean)
        sum_std.append(_std)

len = len(sum_mean)
sum_mean = sum(sum_mean)
sum_std = sum(sum_std)
print (sum_mean,sum_std)
avg_mean = sum_mean/len
avg_std = sum_std/len
print(avg_mean,avg_std)

# touchbottlecaprgbleft avg:mean,std [0.39235646 0.41405597 0.40606222] [0.11579122 0.10381544 0.2021764
# touchbottlecaprgbright avg:mean,std [0.39572367 0.4151846  0.39969215] [0.10486137 0.11412496 0.14097087]
# nontouchrgbleft avg:mean,std [0.39851353 0.42582616 0.4185424 ] [0.12084532 0.10290657 0.20883478]
# nontouchrgbright avg:mean,std [0.4150662  0.44425496 0.4276258 ] [0.09262134 0.11139227 0.14943895]

# all avg:mean,std [0.3949756  0.41411093 0.4044061 ] [0.10875653 0.10813904 0.16938154]