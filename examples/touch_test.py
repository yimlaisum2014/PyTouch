# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import pytouch
import cv2
from PIL import Image
from pytouch.handlers import ImageHandler,SensorHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect


def touch_detect():
    #source = ImageHandler("/path/to/image")
    cv_device = 0
    my_sensor = SensorHandler(cv_device)
    while True:
        frame = my_sensor.get_frame()
        # print(frame)
        cv2.imshow('frame', frame)
        # array to image
        sample = Image.fromarray(frame)
        w,h = sample.size
        print (w,h)

        # initialize with task defaults
        pt = pytouch.PyTouch(DigitSensor, tasks=[TouchDetect])
        is_touching, certainty = pt.TouchDetect(sample)

        print(f"Is touching? {is_touching}, {certainty}")
        c = cv2.waitKey(1)
        if c == 27: break


if __name__ == "__main__":
    touch_detect()
