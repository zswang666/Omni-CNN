import numpy as np
import cv2

#img = bytes array
def decode(img):
	arr = np.fromstring(img, np.uint8)
	im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	return im