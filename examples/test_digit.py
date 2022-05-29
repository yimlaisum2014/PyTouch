from digit_interface.digit import Digit
import pytouch
import cv2
from PIL import Image
from pytouch.handlers import ImageHandler,SensorHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect


def digit_detect():
    #connect the digit device
    digit_right = Digit("D20365", "Right Gripper")
    while True:
        frame = my_sensor.get_frame()
        # print(frame)
        cv2.imshow('frame', frame)
        # array to image
        sample = Image.fromarray(frame)
        w,h = sample.size
        print (w,h)

        c = cv2.waitKey(1)
        if c == 27: break


if __name__ == "__main__":
    digit_detect()