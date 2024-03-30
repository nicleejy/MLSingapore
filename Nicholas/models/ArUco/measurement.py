import cv2
import numpy as np

img = cv2.imread("Nicholas\models\ArUco\phone_aruco_marker.jpg")

# aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
int_corners = np.int0(corners)
cv2.polylines(img, int_corners, isClosed=True, color=(0, 255, 0), thickness=5)

# perimeter of aruco marker = 20cm
aruco_perimeter = cv2.arcLength(int_corners[0], closed=True)

# pixel to cm ratio
pixel_cm_ratio = aruco_perimeter / 20


cv2.imshow("Image", img)
cv2.waitKey(0)
