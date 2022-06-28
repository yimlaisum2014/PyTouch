import os
import sys

import matplotlib.pyplot as plt
import os

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

import pytouch
import cv2
import time
import torch
import torch.nn as nn
from PIL import Image
from digit_interface.digit import Digit
from torchvision import models, transforms
from pytouch.handlers import ImageHandler, SensorHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect
from examples.touch_digit import Filtermap
import traceback
import time
import click


class MyTouchDetectValues:
    SCALES = [64, 64]
    MEANS = [0, 0, 0]
    STDS = [1, 1, 1]
    # MEANS = [0.39534545 ,0.41399255 ,0.40639707]
    # STDS =[0.1094584  ,0.10767335 ,0.17195037]
    CLASSES = 2


class DigitCls(object):

    def __init__(self, sn, name,
                 # model="/home/samyim/PyTouch/train/outputs/2022-05-13/20-08-40/checkpoints/default-epoch=96_val_loss=0.298_val_acc=0.870.ckpt"):
                 model=""):

        self.sn = sn
        self.name = name

        self.model = TouchDetect(model_path=model, defaults=MyTouchDetectValues, sensor=DigitSensor)

        self.connected = False
        self.digit = None

    def connect(self):
        digit_right = Digit(self.sn, self.name)
        while not self.connected:
            print('Try connecting....')
            try:
                digit_right.connect()
                print('Succeed')
                self.connected = True
            except:
                print('Failed')
                pass
                # import traceback
                # print(traceback.format_exc())
            time.sleep(1)

        self.digit = digit_right

    def set_camera(self):
        if self.digit is not None:
            # intensity = (int(5))
            intensity = (int(0))
            # rgb_list = [(intensity, 0, 0), (0, intensity, 0), (0, 0, intensity)]
            # for rgb in rgb_list:
            #     self.digit.set_intensity_rgb(*rgb)
            #     time.sleep(1)
            self.digit.set_intensity_rgb(15, 0, 0)
            # self.digit.set_intensity(intensity)
            # set resolution
            self.digit.set_resolution(Digit.STREAMS['VGA'])
            # set fps
            self.digit.set_fps(15)

    def get_frame(self):
        if self.digit is not None:
            return self.digit.get_frame()

    def detect(self, frame):
        is_touching, certainty = self.model(frame)

        return is_touching, certainty

    def prepare(self):
        self.connect()
        self.set_camera()

    def detect_file(self, image_path):
        img = cv2.imread(image_path)
        frame = Image.fromarray(img)
        is_touching, certainty = self.detect(frame)
        # print(f"{image_path} Is touching? {is_touching}, {certainty}")

        # cv2.imshow('image', img)
        #
        # from torchvision import transforms
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0,0,0], [1,1,1])
        #     # transforms.Normalize([0.0823, 0.0823, 0.0823], [0.1025, 0.1025, 0.1025])
        #     transforms.Normalize([0.3884, 0.4025, 0.3997], [0.1008, 0.0962, 0.1659])
        # ])
        #
        # img_trans = transform(img)
        #
        # mean, std = img_trans.mean([1,2]), img_trans.std([1,2])
        # print(mean, std)
        #
        # arr = img_trans.numpy()
        # max_value = arr.max()
        # arr = arr * 255 / max_value
        # mat = np.uint8(arr)
        # mat = mat.transpose(1, 2, 0)
        # cv2.imshow('mat', mat)
        #
        # cv2.waitKey(0)

        return is_touching

    @staticmethod
    def fft(infile, output):
        img = cv2.imread(infile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fimg = np.fft.fft2(gray)
        fshift = np.fft.fftshift(fimg)

        rows, cols, _ = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0

        ishift = np.fft.ifftshift(fshift)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)

        # cv2.imwrite(output, iimg)
        plt.imsave(output, iimg, cmap='gray')

    @staticmethod
    def get_frames(show=False, save=False, detect=False):
        sn, name = "D20356", "Right Gripper"
        d = DigitCls(sn, name)
        d.prepare()

        idx = 1
        while True:
            frame = d.get_frame()
            if show:
                cv2.imshow("frame", frame)
                cv2.waitKey(0)

            if save:
                filename = f"../train/figure/test_{idx}.png"
                cv2.imwrite(filename, frame)
                click.secho(f'Saved to {filename}', fg='yellow')

            if detect:
                is_touching, certainty = d.detect(Image.fromarray(frame))
                print(f"{idx} Is touching? {is_touching}, {certainty}")

            idx += 1
            time.sleep(1)

    @staticmethod
    def get_classes(root):
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes = sorted(classes)
        class_to_idx = {class_name: x for x, class_name in enumerate(classes)}
        return class_to_idx


def connection():
    digit_right = Digit("D20356", "Right Gripper")
    flag = False
    while not flag:
        print('Try connecting....')
        try:
            digit_right.connect()
            print('Succeed')
            flag = True
        except:
            print('Failed')
            pass
            # import traceback
            # print(traceback.format_exc())
        time.sleep(1)

    intensity = (int(5))
    rgb_list = [(intensity, 0, 0), (0, intensity, 0), (0, 0, intensity)]
    for rgb in rgb_list:
        digit_right.set_intensity_rgb(*rgb)
        time.sleep(1)
    digit_right.set_intensity(intensity)

    frame = digit_right.get_frame()
    return frame


def touch_detect():
    idx = 0
    while True:
        my_custom_model = "../train/outputs/2022-05-12/15-09-32/checkpoints/default-epoch=96_val_loss=0.108_val_acc=0.958.ckpt"
        sample = connection()
        frame = Image.fromarray(sample)
        cv2.imshow('frame', sample)
        cv2.waitKey(0)

        touch_detect = TouchDetect(model=my_custom_model, defaults=MyTouchDetectValues, sensor=DigitSensor)
        is_touching, certainty = touch_detect(frame)
        print(f"{idx} Is touching? {is_touching}, {certainty}")


def featuremap(my_custom_model, frame):
    model = models.resnet18(pretrained=True)
    checkpoint = torch.load(my_custom_model)
    model.load_state_dict(checkpoint, strict=False)
    model_weight = []
    conv_layers = []
    model_children = list(model.children())
    _, conv_layers, _ = Filtermap(model_weight, conv_layers, model_children)

    result = [conv_layers[0](frame)]

    for i in range(1, len(conv_layers)):
        result.append(conv_layers[i](result[-1]))

    output = result

    for num_layer in range(len(output)):
        plt.figure(figsize=(30, 30))
        layer_viz = output[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(f"layer = {layer_viz.size()}")
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='hsv')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"../train/figure/True_layer_{num_layer}.png")


def connect(camera):
    connected = False
    while not connected:
        print('Try connecting....')
        try:
            camera.connect()
            print('Succeed')
            connected = True
        except:
            print('Failed')
            pass
        time.sleep(1)

def prepare_image(img,to_gray):

    if to_gray:
    # gray
        print("Enter gray")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imwrite(f"grayimg_{time.ctime()}.png",gray)
        # make it be 3-channels
        img = np.zeros((gray.shape[0], gray.shape[1], 3))
        img[:,:,0] = gray
        img[:,:,1] = gray
        img[:,:,2] = gray
    # cv2.imwrite(f"after_{time.ctime()}.png",img)
    # to PIL image
    img = PILImage.fromarray(img.astype('uint8'), 'RGB')
    return img

if __name__ == "__main__":
    # touch_detect()
    # DigitCls.get_frames(show=True, save=True, detect=False)
    # sys.exit(1)
    from PIL import Image as PILImage

    sn, name = "D20356", "Right Gripper"
    # digit_right = Digit(sn,name)
    #
    # connect(digit_right)
    #
    # fps = int(1)
    # digit_right.set_fps(fps)


    # without early stop -> overfit
    # model = "/home/sam/PyTouch/train/outputs/2022-05-28/18-24-15/checkpoints/default-epoch=92_val_loss=0.024_val_acc=0.995.ckpt"  # simple CNN  non-touch {1: 618, 0: 582} touch {1: 1196, 0: 4} without transform
    # model = "/home/sam/PyTouch/train/outputs/2022-05-28/19-30-25/checkpoints/default-epoch=96_val_loss=0.333_val_acc=0.862.ckpt"  # Resnet18 non-touch {1: 999, 0: 201} touch {1: 1144, 0: 56} without transform
    # model = "/home/sam/PyTouch/train/outputs/2022-05-28/19-49-49/checkpoints/default-epoch=91_val_loss=0.369_val_acc=0.867.ckpt"  # vgg16 non-touch {1: 216, 0: 984} touch {1: 744, 0: 456}without transform
    # model = "/home/sam/PyTouch/train/outputs/2022-05-28/23-54-14/checkpoints/default-epoch=3_val_loss=0.015_val_acc=0.995.ckpt"  # simple CNN non-touch {1: 0, 0: 4000}, touch {1: 3949, 0: 51} with gray

    # gray image good
    # model = "/home/sam/PyTouch/train/outputs/2022-05-30/22-37-58/checkpoints/default-epoch=1_val_loss=0.125_val_acc=0.969.ckpt" # simple CNN -data /root15 gray bottlecap -class {'nontouchgray': 0, 'touchbottlegray': 1}  -result touch {1: 3741, 0: 259} non-touch {1: 0, 0: 4000}
    # model = "/home/sam/PyTouch/train/outputs/2022-05-30/22-54-30/checkpoints/default-epoch=49_val_loss=0.000_val_acc=1.000.ckpt" #simple CNN -data /root18 rgbbottlecap -class {'nontouchgray': 0, 'touchbottlegray': 1}  -result touch  {1: 0, 0: 4000} non-touch {1: 0, 0: 4000}
    # model = "/home/sam/PyTouch/train/outputs/2022-05-31/03-07-54/checkpoints/default-epoch=49_val_loss=0.000_val_acc=1.000.ckpt" #simple CNN -data /root18 rgbbottlecapwithnorl -class {'nontouchgray': 0, 'touchbottlegray': 1} nontouch {1: 0, 0: 4000} touch {1: 0, 0: 4000}
    # model = "/home/sam/DATA/PyTouch/train/outputs/2022-05-31/04-07-01/checkpoints/default-epoch=9_val_loss=0.075_val_acc=0.988.ckpt" #simple CNN -data /root16 gray cap -class {'nontouchgray': 0, 'touchcapgray': 1} touch {1: 3898, 0: 102} non-touch {1: 3, 0: 3997}
    # model = "/home/sam/DATA/PyTouch/train/outputs/2022-05-31/04-22-27/checkpoints/default-epoch=14_val_loss=0.018_val_acc=0.997.ckpt" #simple CNN -data /root17 gray key -class {'nontouchgray': 0, 'touchkeygray': 1} touch  {1: 3977, 0: 23} non-touch {1: 0, 0: 4000}
    
    # 4 class gray
    # model = "/home/sam/PyTouch/train/outputs/2022-05-31/04-44-54/checkpoints/default-epoch=1_val_loss=0.782_val_acc=0.688.ckpt" #simpleCNN -data /all grey -class {'nontouchgray': 0, 'touchcapgray': 1, 'touchkeygray': 2, 'touchbottlegray': 3}) nontouch {3: 0, 2: 0, 1: 1047, 0: 2953}
                                                                                                                                                                # key {3: 435, 2: 3240, 1: 136, 0: 189} cap {3: 385, 2: 850, 1: 2638, 0: 127} bottle {3: 2104, 2: 1454, 1: 374, 0: 68}
    # rgb different model gray/rgb with/without transform with early stop Both 50 epochs
    # model = "/home/sam/DATA/PyTouch/train/outputs/2022-05-31/05-26-36/checkpoints/default-epoch=49_val_loss=0.029_val_acc=0.999.ckpt" # Resnet18 without transform -data /root17 -class {'nontouchgray': 0, 'touchbottlegray': 1} nontouch  {1: 3, 0: 3997} touch {1: 2798, 0: 1202}
    # model = "/home/sam/PyTouch/train/outputs/2022-05-31/05-45-54/checkpoints/default-epoch=49_val_loss=0.080_val_acc=0.982.ckpt" # Resnet18 without transform -data /root14 -class {'nontouchrgb': 0, 'touchbottlecaprgb': 1} nontouch  {1: 1617, 0: 2383} touch  {1: 38, 0: 3962}
    # model = "/home/sam/PyTouch/train/outputs/2022-05-31/06-08-29/checkpoints/default-epoch=43_val_loss=0.063_val_acc=0.988.ckpt" # Resnet18 transform -data /root14 -class {'nontouchrgb': 0, 'touchbottlecaprgb': 1} nontouch  {1:1919, 0: 2081}  touch  {1: 32, 0: 3968}
    # model = "/home/sam/PyTouch/train/outputs/2022-05-31/06-50-41/checkpoints/default-epoch=20_val_loss=0.142_val_acc=0.972.ckpt"  # VGG16 transform -data /root14 -class {'nontouchrgb': 0, 'touchbottlecaprgb': 1} nontouch 1: 4000, 0: 0} touch   {1:2314, 0: 1686}

    # split model 
    model = "/home/sam/DATA/PyTouch/train/outputs/2022-06-28/14-52-42/checkpoints/default-epoch=49_val_loss=0.040_val_acc=0.989.ckpt" # Simple CNN gray-waterbottle Left
    # model = "/home/sam/DATA/PyTouch/train/outputs/2022-06-27/05-15-01/checkpoints/default-epoch=98_val_loss=0.002_val_acc=1.000.ckpt" # Resenet18 gray-waterbottle right
    
    # model = TouchDetect(model_path=model, defaults=MyTouchDetectValues, sensor=DigitSensor)
    # count = 0
    # import pandas as pd
    #
    # df = pd.DataFrame()
    #
    # for n in range(10):
    #     print("before",time.ctime())
    #     right_image = digit_right.get_frame()
    #     print("after",time.ctime())
    #
    #     # image_serial = pd.Series(data=[right_image[0],right_image[1]],index=["W","H"])
    #     # df = pd.DataFrame()
    #     # df.append(image_serial, ignore_index=True)
    #
    #     img = right_image.reshape((1, -1))
    #     predict_image = prepare_image(right_image, to_gray=False)
    #     # result, _ = model(predict_image)
    #
    #     row_df = pd.concat([pd.DataFrame([n], columns=['id']), pd.DataFrame([time.ctime()], columns=['data']), pd.DataFrame(img)], axis=1)
    #     df = pd.concat([df, row_df])
    #
    # df = df.set_index('id')
    # print(df)

    d = DigitCls(sn, name, model=model)

    # file = "/home/sam/Dataset/modify/root8/non-touch/_90/left_74.png"  # False: {1: 1365, 0: 635}
    # file = "/home/sam/Dataset/modify/root8/touch/_60/processed_img2_flipHorizontal_trial_3_65.png"  # True:   {1: 1128, 0: 448}
    

    # gray root 15 bottle 16 cap 17 key

    # file = "/home/sam/DATA/Dataset/modify/root15/nontouchgray"  # False:
    # file = "/home/sam/DATA/Dataset/modify/root17/touchkeygray"  # True:
    # file = "/home/sam/Dataset/modify/root16/touchcapgray"  # True:
    #file = "/home/sam/Dataset/modify/root15/touchbottlegray"  # True:

    # rgb root 14 bottle
    # file = "/home/sam/Dataset/modify/root14/nontouchrgb"  # False:
    # file = "/home/sam/Dataset/modify/root14/touchbottlecaprgb"  # True:

    # test-data folder
    # file = "/home/sam/digit-interface/example/root20/testbottlegray" # simplecnn-gray-bottle {1: 763, 0: 1237}  restnet18-gray-bottle  {1: 1441, 0: 559 }
    # file = "/home/sam/digit-interface/example/root20/testcapgray" # simplecnn-gray-cap   {1: 794, 0: 1206}
    # file = "/home/sam/digit-interface/example/root20/testkeygray" # simplecnn-gray-key  {1: 1470, 0: 530}
    # file = "/home/sam/digit-interface/example/root20/testnontouchgray" # simplecnn-gray-nontouch key {0, 1: 0, 0: 2000} cap {1: 267, 0: 1733} restnet18-gray-nontouch  {1: 308, 0: 1692}

    # split data
    file = "/home/sam/DATA/Dataset/modify/left-bottlecap-gray/nontouchgray" # Simple {touch 1: 2000, non-touch 0: 0}
    # file = "/home/sam/DATA/Dataset/modify/left-bottlecap-gray/touchbottlecapgray" # {touch 1: 38, non-touch 0: 1962}
    # file = "/home/sam/DATA/Dataset/modify/left-cap-gray/nontouchgray" #
    # file = "/home/sam/DATA/Dataset/modify/left-cap-gray/touchcapgray" #
    # file = "/home/sam/DATA/Dataset/modify/right-bottlecap-gray/touchbottlecapgray" #
    # file = "/home/sam/DATA/Dataset/modify/right-bottlecap-gray/nontouchgray" #
    # file = "/home/sam/DATA/Dataset/modify/right-bottlecap-gray/touchbottlecapgray" #



    # image_path = file
    # feature_model = "../train/outputs/2022-05-12/15-09-32/checkpoints/default-epoch=96_val_loss=0.108_val_acc=0.958.ckpt"
    #
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),root14
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    # ])
    #
    # frame = cv2.imread(image_path)
    # frame = transform(frame)
    # frame = frame.unsqueeze(0)
    # featuremap(feature_model, frame)

    # print class and index
    #####################################################################################
    print('class and index map')
    root = os.path.dirname(file)
    print(d.get_classes(root))


    import os

    # folder = os.path.abspath(os.path.dirname(file))

    # count = {1: 0, 0: 0}
    count = {3:0, 2:0,1: 0, 0: 0}

    for root, dirs, names in os.walk(file):

        with click.progressbar(names, length=len(names)) as names:
            for name in names:
                filename = os.path.join(root, name)
                value = d.detect_file(filename)
                count[value] += 1

    print(count)
    #######################################################################################
    # file = "/home/sam/Dataset/modify/root7/non-touch/_90"  # False: {1: 904, 0: 296} Resnet18
    # file = "/home/sam/Dataset/modify/root7/touch/_60"  # True: {1: 845, 0: 355} Resnet18
    #
    # pt = pytouch.PyTouch(DigitSensor, tasks=[TouchDetect])
    # # touch_detect = TouchDetect(DigitSensor, zoo_model="touchdetect_resnet18")
    #
    # folder = os.path.abspath(os.path.dirname(file))
    #
    # count = {1: 0, 0: 0}
    #
    # for root, dirs, names in os.walk(folder):
    #     for name in names:
    #         filename = os.path.join(root, name)
    #         value, _ = pt.TouchDetect(Image.fromarray(cv2.imread(filename)))
    #         count[value] += 1
    #
    # print(count)
