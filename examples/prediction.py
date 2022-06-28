import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray 
from cv_bridge import CvBridge
from pytouch.tasks import TouchDetect
from pytouch.sensors import DigitSensor

class DetectectValue:
    SCALES = [240, 320]
    MEANS = [0, 0, 0]
    STDS = [1, 1, 1]
    CLASSES = 2

class Prediction():
    def __init__(self,model_path):

        self.bridge = CvBridge()
        # Subscriber
        self.sub_left_digit_image = rospy.Subscriber("finger_left",Image,self.left_digit_image_callback)
        self.sub_right_digit_image = rospy.Subscriber("finger_right",Image,self.right_digit_image_callback)
        
        # Defin Model
        self.model = TouchDetect(model_path=model_path, defaults=DetectectValue, sensor=DigitSensor)
        self.results = [0,0]
        
        # Publisher 
        self.pub_result = rospy.Publisher('predict_result', Int16MultiArray  , queue_size=10)
        self.pub_result.publish(self.results)

    def left_digit_image_callback(self,msg):
        left = msg.data
        predict = self.prediction(left)
        self.results[0] = predict
        
    
    def right_digit_image_callback(self,msg):
        right = msg.data
        predict = self.prediction(right)
        self.results[1] = predict
        
    
    def prediction(self,image_msg):
        predict_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        result,_ = self.model(predict_image)
        return result

if __name__():
    rospy.init_node('prediction', anonymous=True)
    model_path = ""
    prediction = Prediction(model_path)
    rospy.spin()


