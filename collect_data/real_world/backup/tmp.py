import json
import numpy as np
import cv2
from copy import deepcopy
from math import pi
from PIL import Image, ImageTk
from scipy.optimize import minimize
import os
import to_equi
import pdb

root_dir = "/home/joehuang/Desktop/Final_Test_Real_bk/demonstration/EECS7-fish/room_711"
image_fns = os.listdir(root_dir)
path_orders = [int(image_fn[:-4]) for image_fn in image_fns]
path_orders = np.sort(np.array(path_orders))
image_fns = [str(path_order) + ".jpg" for path_order in path_orders]
print image_fns
pdb.set_trace()

save_dir = './tmp/'
image_count = 0
for image_fn in image_fns:
    image_path = os.path.join(root_dir, image_fn)
    save_fn = str(image_count) + '.jpg'
    save_path = os.path.join(save_dir, save_fn)

    a = 1280
    lens1 = to_equi.FisheyeLens()
    lens2 = to_equi.FisheyeLens()
    [data1 , data2] = [{"cf": 190.0, "cr": 480.0, "cx": 480, "cy": 540, "qw": 1, "qx": 0, "qy": 0, "qz": 0},
    {"cf": 190.0, "cr": 480.0, "cx": 1440, "cy": 540, "qw": 0, "qx": 0, "qy": 1, "qz": 0}]
    lens1.from_dict(data1)
    lens2.from_dict(data2)
    image = cv2.imread(image_path)

    img1 = to_equi.FisheyeImage(image,lens1)
    img2 = to_equi.FisheyeImage(image,lens2)
    pan = to_equi.PanoramaImage((img1, img2))
    equi = pan.render_equirectangular(a)
    cvimage = pan.render_cubemap(a)

    #xp = cvimage[0:a, :, :]
    #xp = cv2.flip(xp, 1)
    #cv2.imwrite(save_path,xp)
    cv2.imwrite(save_path,equi)
    image_count += 1
    print image_count
