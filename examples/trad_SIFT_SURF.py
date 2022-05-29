import cv2
import numpy as np
import matplotlib.pyplot as plt

file1 = "/home/sam/Dataset/modify/root7/non-touch/trial_4/left_50.png" #False
file2 = "/home/sam/Dataset/modify/root7/touch/_55/left_50_2.png"  #True

img1 = cv2.imread(file1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread(file2, cv2.COLOR_BGR2RGB)
print (img2.shape)

# # split three channel
# (B, G, R) = cv2.split(img2)
# zeros = np.zeros(img2.shape[:2], dtype="uint8")
# # cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
# # cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
# # cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

# SIFT method
sift = cv2.xfeatures2d.SIFT_create()
# surf = cv2.xfeatures2d.SURF_create()
# orb = cv2.ORB_create(nfeatures=1500)
keypoints_sift, descriptors = sift.detectAndCompute(img1, None)
# keypoints_surf, descriptors = surf.detectAndCompute(img2, None)
# keypoints_orb, descriptors = orb.detectAndCompute(img1, None)

print(len(keypoints_sift), descriptors)
img = cv2.drawKeypoints(img1, keypoints_sift, None)
cv2.imshow("Image", img)

cv2.imshow("True", img2)
cv2.imshow("False", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("END")