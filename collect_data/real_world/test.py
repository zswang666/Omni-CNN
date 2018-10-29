import json
import numpy as np
import cv2
from copy import deepcopy
from math import pi
from PIL import Image, ImageTk
from scipy.optimize import minimize
import os
import to_equi


if __name__ == "__main__":
    a = 1280
    lens1 = to_equi.FisheyeLens()
    lens2 = to_equi.FisheyeLens()
    [data1 , data2] = [{"cf": 190.0, "cr": 480.0, "cx": 480, "cy": 540, "qw": 1, "qx": 0, "qy": 0, "qz": 0},
    {"cf": 190.0, "cr": 480.0, "cx": 1440, "cy": 540, "qw": 0, "qx": 0, "qy": 1, "qz": 0}]
    lens1.from_dict(data1)
    lens2.from_dict(data2)
    image = cv2.imread('./test_images/test.jpg')

    img1 = to_equi.FisheyeImage(image,lens1)
    img2 = to_equi.FisheyeImage(image,lens2)
    pan = to_equi.PanoramaImage((img1, img2))
    equi = pan.render_equirectangular(a)
    cvimage = pan.render_cubemap(a)

    xp = cvimage[0:a, :, :]
    xm = cvimage[a:2*a, :, :]
    yp = cvimage[2*a:3*a, :, :]
    ym = cvimage[3*a:4*a, :, :]

    xp = cv2.flip(xp, 1)
    xm = cv2.flip(xm, 1)
    yp = cv2.flip(yp, 1)
    ym = cv2.flip(ym, 1)

    cv2.imwrite('equi.jpg',equi)
    cv2.imwrite('front.jpg',xp)
    cv2.imwrite('back.jpg',xm)
    cv2.imwrite('right.jpg',yp)
    cv2.imwrite('left.jpg',ym)
