# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import pytouch
import cv2
import time
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

class camThread(threading.Thread):
    def __init__(self, previewName ,devSerial):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.devSerical = devSerial
        

    def run(self):
        print("starting " + self.previewName)
        camPreview(self.previewName, self.devSerical)


def connect(devSerical,previewName,trigger):
    
    digit_dev = Digit(devSerical, previewName)

    if trigger == True :
        digit_dev.disconnect()

    else:
        digit_dev.connect()
        print("connected")
        intensity = int(5)
        # Maximum value for each channel is 15
        rgb_list = [(intensity, 0, 0), (0, intensity, 0), (0, 0, intensity)]
        for rgb in rgb_list:
            digit_dev.set_intensity_rgb(*rgb)
            time.sleep(1)
        digit_dev.set_intensity(intensity)
        time.sleep(1)
        print("setted intensity")
        
        frame = digit_dev.get_frame()       
        return frame

def camPreview(previewName, devSerical):
    print(previewName, devSerical)

    result = connect(devSerical,previewName,False)

    if previewName == "Right Gripper":
        print("left")
        
        cv2.imshow(previewName, result)
        key = cv2.waitKey(1)
        if key == 27:  # exit on ES
            connect(devSerical,previewName,True)
        
    if previewName == "Left Gripper":
        print("right")
        
        cv2.imshow(previewName, result)
        key = cv2.waitKey(1)
        if key == 27:  # exit on ES
            connect(devSerical,previewName,True)
            

        # while result is not None:
        #     cv2.imshow(previewName, result)
        #     print(result)
        
        # while True:
        #     print("hello")
        #     cv2.namedWindow(previewName)

        #     if result is not None:  # try to get the first frame
        #         rval = True
        #         frame = result
        #     else:
        #         rval = False
        #     print(rval)

        #     while rval:
        #         cv2.imshow(previewName, frame)
                
        #         key = cv2.waitKey(1)
        #         if key == 27:  # exit on ESC
        #             break
        #     cv2.destroyWindow(previewName)

def touch_detect():

    cv_device_left = 0
    cv_device_right = 1

    my_sensor_left = SensorHandler(cv_device_left)
    my_sensor_right = SensorHandler(cv_device_right)
    while True:
        frame_left = my_sensor_left.get_frame()
        frame_right = my_sensor_right.get_frame()

        # print(frame)
        cv2.imshow('frame_left', frame_left)
        cv2.imshow('frame_right', frame_right)
        # array to image
        sample_left = Image.fromarray(frame_left)
        sample_right = Image.fromarray(frame_right)
        #w,h = sample_left.size
        #print (w,h)

        # initialize with custom configuration of TouchDetect task
        my_custom_model = "~/PyTouch/train/outputs/2022-05-01/17-12-15/checkpoints/name:default-epoch=37_val_loss=0.058_val_acc=0.987.ckpt"
        touch_detect = TouchDetect(model=my_custom_model, defaults=MyTouchDetectValues, sensor= DigitSensor)
        is_touching_left, certainty_left = touch_detect(sample_left)
        is_touching_right, certainty_right = touch_detect(sample_right)
        print(f"left,right? {is_touching_left}, {certainty_left}, {is_touching_right}, {certainty_right}")

        c = cv2.waitKey(1)
        if c == 27: break

    # source = ImageHandler(img_path = "../../Dataset/local/datasets/real/sphere/digit-0.5mm-ball-bearing-zup_2021-08-31-21-37-35/test/0002/color/0080.png")
    # my_custom_model = "~/PyTouch/train/outputs/2022-05-01/17-12-15/checkpoints/name:default-epoch=37_val_loss=0.058_val_acc=0.987.ckpt"

    # # initialize with custom configuration of TouchDetect task
    # touch_detect = TouchDetect(model=my_custom_model, defaults=MyTouchDetectValues, sensor= DigitSensor)
    # is_touching, certainty = touch_detect(source.image)
    # print(f"Is touching? {is_touching}, {certainty}")


if __name__ == "__main__":

    thread1 = camThread("Right Gripper", "D20356")
    thread2 = camThread("Left Gripper", "D20365")
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    #print("Active threads", threading.activeCount())
    # touch_detect()
