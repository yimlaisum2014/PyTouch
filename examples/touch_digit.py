# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import os

import matplotlib.pyplot as plt

# import pytouch
import cv2
import time
import torch
import torch.nn as nn
from PIL import Image
from digit_interface.digit import Digit
from torchvision import models, transforms
# from pytouch.handlers import ImageHandler,SensorHandler
# from pytouch.sensors import DigitSensor
# from pytouch.tasks import TouchDetect

import traceback
import time

class MyTouchDetectValues:
    SCALES = [128, 128]
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
    CLASSES = 2

def connection():
    digit_right = Digit("D20356", "Right Gripper")
    flag = False
    while not flag:
        print('Try connecting....')
        try :
            digit_right.connect()
            print('Succeed')
            flag = True
        except:
            print('Failed')
            pass
            # import traceback
            # print(traceback.format_exc())
        time.sleep(1)


    intensity= (int(5))
    rgb_list = [(intensity, 0, 0), (0, intensity, 0), (0, 0, intensity)]
    for rgb in rgb_list:
        digit_right.set_intensity_rgb(*rgb)
        time.sleep(1)
    digit_right.set_intensity(intensity)

    frame = digit_right.get_frame()
    return frame

def Filtermap(model_weight,conv_layers,model_children):

    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weight.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weight.append((child.weight))
                        conv_layers.append(child)
    print(f"Total convolution layer: {counter}")

    for weight, conv in zip(model_weight, conv_layers):
        print(f"CON: {conv} ==> SHAPE: {weight.shape}")

    plt.figure(figsize=(20, 17))
    print(type(model_children))
    print(type(model_children[0]))
    for i, filter in enumerate(model_weight[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter[0, :, :].detach().numpy(), cmap="hsv")
        plt.axis("off")
        plt.savefig("../train/figure/filter.png")
    plt.show(block=False)
    return (model_weight,conv_layers,model_children)

def touch_detect():
    #source = ImageHandler("/path/to/image")
    #cv_device = 0
    #my_sensor = SensorHandler(cv_device)
    
    # frame = connection()
    # # print(frame)
    # cv2.imshow('frame', frame)
    # # array to image
    # sample = Image.fromarray(frame)
    # w, h = sample.size
    # print(w, h)

    # while True:

        # initialize with task defaults
        # pt = pytouch.PyTouch(DigitSensor, tasks=[TouchDetect])
        # is_touching, certainty = pt.TouchDetect(sample)
        # print(f"Is touching? {is_touching}, {certainty}")

        # initialize with custom configuration of TouchDetect task
        # source = os.listdir()
        # print(source)
        my_custom_model = "../train/outputs/2022-05-12/15-09-32/checkpoints/default-epoch=96_val_loss=0.108_val_acc=0.958.ckpt"

        model = models.resnet18(pretrained=True)
        print(model)
        # checkpoint = model.load_from_checkpoint(my_custom_model)
        # checkpoint = torch.load(my_custom_model)
        # # print(checkpoint)
        # model.load_state_dict(checkpoint, strict=False)
        #
        # model_weight = []
        # conv_layers = []
        # model_children = list(model.children())
        #
        # # Take a look of filtermap
        # _, conv_layers, _ = Filtermap(model_weight,conv_layers,model_children)
        #
        # frame = connection()
        # print(type(frame),frame)
        #
        #
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((128, 128)),
        #     transforms.ToTensor(),
        # ])
        #
        #
        # frame = transform(frame)
        # frame = frame.unsqueeze(0)
        # result = [conv_layers[0](frame)]
        #
        # for i in range(1, len(conv_layers)):
        #     result.append(conv_layers[i](result[-1]))
        #
        # output = result
        # print(len(output))
        #
        # for num_layer in range(len(output)):
        #     plt.figure(figsize=(30, 30))
        #     layer_viz = output[num_layer][0, :, :, :]
        #     layer_viz = layer_viz.data
        #     print(f"layer = {layer_viz.size()}")
        #     for i, filter in enumerate(layer_viz):
        #         if i == 64:  # we will visualize only 8x8 blocks from each layer
        #             break
        #         plt.subplot(8, 8, i + 1)
        #         plt.imshow(filter, cmap='hsv')
        #         plt.axis("off")
        #     print(f"Saving layer {num_layer} feature maps...")
        #     plt.savefig(f"../train/figure/layer_{num_layer}.png")
        #     # plt.show()
        #     plt.close()



        # touch_detect = TouchDetect(model=my_custom_model, defaults=MyTouchDetectValues, sensor= DigitSensor)
        # is_touching, certainty = touch_detect(sample)
        # print(f"Is touching? {is_touching}, {certainty}")

        # c = cv2.waitKey(1)
        # if c == 27: break


if __name__ == "__main__":
    touch_detect()
