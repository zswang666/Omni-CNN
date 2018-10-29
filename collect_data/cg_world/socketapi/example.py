import GameController as gc
import imgdecode as imd
import numpy as np
import cv2


con = gc.Controller()
con.connect()

keyarray = np.zeros(128, dtype=np.int)

while True:
	img = con.getDepth()
	img = imd.decode(img)
	cv2.imshow('Depth', img)
	
	key = cv2.waitKey(10)
	if key <= 127:
		keyarray[key] = ~keyarray[key]
		ckey = chr(key)
		if ckey.isalnum():
			if keyarray[key] != 0:
				con.KeyDown(ckey)
			else:
				con.KeyUp(ckey)
		elif key == 27:
			break;



con.close()