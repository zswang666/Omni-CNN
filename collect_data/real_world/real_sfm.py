import math
import os
import numpy as np
import pdb
import cv2
import json

def robot_rotation(shot):
    R = cv2.Rodrigues(np.array(shot['rotation'], dtype=float))[0]
    t = np.array([0, 0, 1])
    view_direction = R.T.dot(t)
    x = view_direction[0]
    y = view_direction[1]
    angle = math.atan2(x,y)
    return angle * 180 / math.pi

def optical_center(shot):
    R = cv2.Rodrigues(np.array(shot['rotation'], dtype=float))[0]
    t = shot['translation']
    return -R.T.dot(t)

def run_sfm(scene_name, filename_list):
    """
    This is the function to connect to opensfm
    given scene name and list of full paths, return a dict of string to tuple of np.1darray
    the key of dict is string (image path) and value is tuple of np.1darray (np.array([x,y,z]), np.array([ax,ay,az]))
    NOTE!: in order to not run opensfm for same scene over one time, please save the sfm result for every scene using scene tag
            when user try to sfm the same scene again, ask them whether to rerun opensfm or use old result to save time
    Input:
        scene_name: string; the name of the scene, it is also the unique tag for the scene
        path_list: list of string; list of full path of the input images to opensfm, the images are all W=1280, H=768, D=3
    Return:
        image_dict: string to tuple of np.1darray;
            keys: image_path: string; the given path
            values: image_location: tuple of np.1darray; the location (position, rotation) information of the image
    """
    # assume sfm has done, read sfm file and output the best image list
    reconstruct_path = raw_input('>>> input the reconstructed json path, press d to use default: /home/joehuang/Desktop/Final_Test_Real/eecs_reconstruction.json: ')
    if reconstruct_path == 'd':
        reconstruct_path = '/home/joehuang/Desktop/Final_Test_Real/eecs_reconstruction.json'
    delete_names = [str(num) + '.jpg' for num in range(607,623)]

    scale = raw_input('>>> input the scale in float: ')
    scale = float(scale)
    with open(reconstruct_path) as f:
        reconstruct_data = json.load(f)
    construct_data = reconstruct_data[0]['shots']
    image_names = construct_data.keys()

    # get related path
    name2filename = {} #000.jpg -> /home/...
    for image_filename in filename_list:
        image_id = int(image_filename[:-4])
        name = str("{:03}".format(image_id)) + ".jpg"
        name2filename[name] = image_filename

    image_dict = {}
    for image_name in image_names:
        shot = construct_data[image_name]
        if image_name in delete_names:
            print image_name
            continue
        if image_name in name2filename.keys():
            image_filename = name2filename[image_name]
            angle = robot_rotation(shot)
            position = optical_center(shot)
            angle = np.array([0., angle, 0.])
            position = np.array([position[0], position[2], position[1]]) / scale
            image_dict[image_filename] = (position, angle)
        else:
            print "luna failed image"

    return image_dict
    # print sfm failed images
    raise NotImplementedError("not implemented")

