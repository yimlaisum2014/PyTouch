# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import pytouch
import cv2
import threading
from PIL import Image
from digit_interface.digit import Digit

from pytouch.handlers import ImageHandler,SensorHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect
from pytouch.models.touch_detect import TouchDetectModelDefaults

class MyTouchDetectValues:
    SCALES = [64, 64]
    MEANS = [0.123, 0.123, 0.123]
    STDS = [0.123, 0.123, 0.123]
    CLASSES = 2

def touch_detect():

    source = ImageHandler(img_path = "../../Dataset/modify/root8/touch/_55/processed_img1_flipHorizontal_trial_1_1.png")
    #my_custom_model = "~/PyTouch/train/outputs/2022-05-01/17-12-15/checkpoints/name:default-epoch=37_val_loss=0.058_val_acc=0.987.ckpt"
    x = source.image
    w,h = x.size
    print (w,h)
    # initialize with custom configuration of TouchDetect task
    #touch_detect = TouchDetect(model=my_custom_model, defaults=MyTouchDetectValues, sensor= DigitSensor)
    #is_touching, certainty = touch_detect(source.image)
    #print(f"Is touching? {is_touching}, {certainty}")


if __name__ == "__main__":
    touch_detect()
