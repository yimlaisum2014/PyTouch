import pytouch
import cv2
import time
import threading

from PIL import Image
from digit_interface.digit import Digit

from multiprocessing import Process
from pytouch.handlers import ImageHandler,SensorHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect
from pytouch.models.touch_detect import TouchDetectModelDefaults

class MyTouchDetectValues:
    SCALES = [64, 64]
    MEANS = [0.123, 0.123, 0.123]
    STDS = [0.123, 0.123, 0.123]
    CLASSES = 2

class camProcess(Process):
    def __init__(self, camName ,devSerial):
        Process.__init__(self)
        self.camName = camName
        self.devSerical = devSerial


    def run(self):
        print("starting " + self.camName)
        camConnection(self.camName, self.devSerical)

def instensity(digit_dev,value):

    rgb_list = [(value, 0, 0), (0, value, 0), (0, 0, value)]
    for rgb in rgb_list:
        digit_dev.set_intensity_rgb(*rgb)
        time.sleep(1)
    digit_dev.set_intensity(value)
    time.sleep(1)
    print("connected")
    
    frame = digit_dev.get_frame()       
    return frame

def camConnection(name, devSerial):
    digit_dev = Digit(devSerial, name)
    digit_dev.connect()
    instensity(digit_dev,int(5))
    while True : 
        frame = digit_dev.get_frame()
        detection(name,frame)
        cv2.imshow(name, frame)
        key = cv2.waitKey(1)
        if key == 27:  # exit on ES
            break
    digit_dev.disconnect()

def detection(name,frame):
    sample = Image.fromarray(frame)
    my_custom_model = "~/PyTouch/train/outputs/2022-05-09/13-51-01/checkpoints/default-epoch=87_val_loss=0.651_val_acc=0.833.ckpt"
    touch_detect = TouchDetect(model=my_custom_model, defaults=MyTouchDetectValues, sensor= DigitSensor)
    is_touching, certainty = touch_detect(sample)
    print(f"{name},{is_touching}, {certainty}")

if __name__ == "__main__":

    Process_1 = camProcess("Right Gripper", "D20356")
    Process_2 = camProcess("Left Gripper", "D20365")
    Process_1.start()
    Process_2.start()
    Process_1.join()
    Process_2.join()
    #print("Active threads", threading.activeCount())
    # touch_detect()
