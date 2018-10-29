import sys
import json
import cv2
import numpy as np
from copy import deepcopy
from math import pi
from PIL import Image, ImageTk
from scipy.optimize import minimize
import os
import to_equi
import real_sfm
import pdb

"""
this function take in a root directory of fish eye raw image and packed them into demonstration dataset
it can transform both omni dataset and quad dataset
omni dataset style: json
    scene_name: string
    time_step: float, will all be -1.0
    time: list of float, will all be -1.0
    position: list of list of float (x,y,z)
    angle: list of list of float (x,y,z)
    semantic_tag: list of string; the room tag
    image_path: list of string; list of image paths
quad dataset style: json
    scene_name: string
    time_step: float, will all be -1.0
    time: list of float, will all be -1.0
    position: list of list of float (x,y,z)
    angle: list of list of float (x,y,z)
    semantic_tag: list of string; the room tag
    image_path_quad: list of list of string (length 4); list of image path quad
"""

if __name__ == '__main__':
    # ask for root directory
    root_dir = raw_input('>>> Final realworld data root directory (enter "d" to use default: /home/joehuang/Desktop/Final_Test_Real): ')
    if root_dir == 'd':
        root_dir = '/home/joehuang/Desktop/Final_Test_Real'
    demonstration_dir = os.path.join(root_dir, 'demonstration')

    # ask for scene name, ex: EECS, look for EECS_fish, raise if file not exists
    scene_name = raw_input('>>> please input the name of the scene, ex: EECS7 (enter "d" to use default: EECS7): ')
    if scene_name == 'd':
        scene_name = 'EECS7'
    fisheye_dir = os.path.join(demonstration_dir, scene_name + '-fish')
    if not os.path.exists(fisheye_dir):
        print("real world fisheye data file {} not found".format(fisheye_dir))
        print("Program Quit!")
        sys.exit()

    # check scene + '_converted.txt' exists or not, ex: EECS_converted.txt
    omni_converted_collection_filename = scene_name + '_converted_omni.json'
    omni_converted_collection_path = os.path.join(root_dir, omni_converted_collection_filename)
    quad_converted_collection_filename = scene_name + '_converted_quad.json'
    quad_converted_collection_path = os.path.join(root_dir, quad_converted_collection_filename)

    omni_location_bag_filename = scene_name + '_omni.json'
    omni_location_bag_path = os.path.join(root_dir, omni_location_bag_filename)
    quad_location_bag_filename = scene_name + '_quad.json'
    quad_location_bag_path = os.path.join(root_dir, quad_location_bag_filename)
    if os.path.exists(omni_location_bag_path):
        print("the location bag {} and {} already exists! Please delete it if you want to recreate the location bag file".format(omni_location_bag_path, quad_location_bag_path))
        sys.exit()
    convert_fisheye_images_flag = True
    if os.path.exists(omni_converted_collection_path):
        print("converted collection file is found")
        convert_fisheye_images = raw_input('>>> do you want to re-convert fisheye images to equi-rectangular and cubic images? (y/n)')
        if convert_fisheye_images == 'y':
            print("you should delete original transfered images and convert collection files then run this file again")
            print("Program Quit!")
            sys.exit()
        elif convert_fisheye_images == 'n':
            convert_fisheye_images_flag = False
        else:
            raise ValueError("unknown flag, please answer n or y")


    if convert_fisheye_images_flag:
        print("convert fisheye images to equirectangular images and quadrupal monocular images")
        omni_data = {}
        omni_data['scene_name'] = scene_name
        omni_data['time_step'] = -1.
        omni_data['time'] = []
        omni_data['position'] = []
        omni_data['angle'] = []
        omni_data['semantic_tag'] = []
        omni_data['image_path'] = []

        quad_data = {}
        quad_data['scene_name'] = scene_name
        quad_data['time_step'] = -1.
        quad_data['time'] = []
        quad_data['position'] = []
        quad_data['angle'] = []
        quad_data['semantic_tag'] = []
        quad_data['image_path_quad'] = []

        print("collect all fisheye paths, and define all quad image paths, and equi image paths")
        quad_subdir = scene_name + '-quad'
        quad_dir = os.path.join(demonstration_dir, quad_subdir)
        omni_subdir = scene_name + '-omni'
        omni_dir = os.path.join(demonstration_dir, omni_subdir)
        print("create both omni-directional image directory {} and quad-monocular image directory {}".format(omni_dir, quad_dir))
        os.mkdir(omni_dir)
        os.mkdir(quad_dir)
        image_count = 0
        for room_subdir in os.listdir(fisheye_dir):
            room_id = room_subdir #also semantic tag
            room_dir = os.path.join(fisheye_dir, room_subdir)
            for image_filename in os.listdir(room_dir):
                fish_image_path = os.path.join(room_dir, image_filename)
                equi_image_filename = str(image_count) + '.jpg'
                front_image_filename = str(image_count) + '_1.jpg'
                right_image_filename = str(image_count) + '_2.jpg'
                back_image_filename = str(image_count) + '_3.jpg'
                left_image_filename = str(image_count) + '_4.jpg'

                equi_image_path = os.path.join(omni_dir, equi_image_filename)
                front_image_path = os.path.join(quad_dir, front_image_filename)
                right_image_path = os.path.join(quad_dir, right_image_filename)
                back_image_path = os.path.join(quad_dir, back_image_filename)
                left_image_path = os.path.join(quad_dir, left_image_filename)

                width = 1280
                lens1 = to_equi.FisheyeLens()
                lens2 = to_equi.FisheyeLens()
                [data1 , data2] = [{"cf": 190.0, "cr": 480.0, "cx": 480, "cy": 540, "qw": 1, "qx": 0, "qy": 0, "qz": 0},
                {"cf": 190.0, "cr": 480.0, "cx": 1440, "cy": 540, "qw": 0, "qx": 0, "qy": 1, "qz": 0}]
                lens1.from_dict(data1)
                lens2.from_dict(data2)
                image = cv2.imread(fish_image_path)

                img1 = to_equi.FisheyeImage(image,lens1)
                img2 = to_equi.FisheyeImage(image,lens2)
                pan = to_equi.PanoramaImage((img1, img2))
                equi = pan.render_equirectangular(width)
                cvimage = pan.render_cubemap(width)

                front = cvimage[0:width, :, :]
                back = cvimage[width:2*width, :, :]
                right = cvimage[2*width:3*width, :, :]
                left = cvimage[3*width:4*width, :, :]

                front = cv2.flip(front, 1)
                back = cv2.flip(back, 1)
                right = cv2.flip(right, 1)
                left = cv2.flip(left, 1)

                front = front[256:1024,:,:]
                back = back[256:1024,:,:]
                right = right[256:1024,:,:]
                left = left[256:1024,:,:]

                cv2.imwrite(equi_image_path, equi)
                cv2.imwrite(front_image_path, front)
                cv2.imwrite(back_image_path, back)
                cv2.imwrite(left_image_path, left)
                cv2.imwrite(right_image_path, right)

                omni_data['semantic_tag'].append(room_id)
                omni_data['time'].append(-1.0)
                omni_data['image_path'].append(equi_image_path)

                quad_data['semantic_tag'].append(room_id)
                quad_data['time'].append(-1.0)
                quad_data['image_path_quad'].append([front_image_path, right_image_path, back_image_path, left_image_path])
                image_count += 1
                print("successfully processed {} images".format(image_count))
        with open(omni_converted_collection_path, 'w') as f:
            json.dump(omni_data, f)
        print("successfully saved omni converted collection file in {}".format(omni_converted_collection_path))
        with open(quad_converted_collection_path, 'w') as f:
            json.dump(quad_data, f)
        print("successfully saved quad converted collection file in {}".format(quad_converted_collection_path))
    else:
        print("use old equirectangular images and quadrupal monocular images")
        with open(omni_converted_collection_path) as f:
            omni_data = json.load(f)
        with open(quad_converted_collection_path) as f:
            quad_data = json.load(f)

    # we only use quad data for sfm
    image_path_quad_list = quad_data['image_path_quad']
    image_path_list = []
    path2indices = {}
    for image_index in range(len(image_path_quad_list)):
        image_path_quad = image_path_quad_list[image_index]
        for quad_index in range(4):
            image_path = image_path_quad[quad_index]
            image_path_list.append(image_path)
            path2indices[image_path] = (image_index, quad_index)
    pdb.set_trace()
    print("\nstart running opensfm to get demonstration image locations\n")
    path2location = real_sfm.run_sfm(scene_name, image_path_list)
    print("\nsuccessfully running sfm and convert into {}/{} positions/paths\n".format(len(path2location), len(image_path_quad_list)))
    # since not all input image path have corresponding output location, I will have to filter out failed images
    # any images in quad successfully localize is a sucessful demonstration set
    position_list = []
    angle_list = []
    quad_index_list = [] # the quad index of this location to have sfm generated location
    for index in range(len(image_path_quad_list)):
        # initialized as some false words
        angle_list.append(None)
        position_list.append(None)
        quad_index_list.append(-1)

    for image_path in path2location.keys():
        position_np, angle_np = path2location[image_path]
        image_index, quad_index = path2indices[image_path]
        last_best_quad_index = quad_index_list[image_index]
        record_location = False
        if last_best_quad_index == -1: # -1 (no previous recorded), then record
            record_location = True
        elif last_best_quad_index in [0,2]: # previous is front or back
            record_location = False
        else:
            record_location = True

        if record_location:
            angle_list[image_index] = list(angle_np)
            position_list[image_index] = list(position_np)
            quad_index_list[image_index] = quad_index

    omni_dem_data = {}
    omni_dem_data['scene_name'] = omni_data['scene_name']
    omni_dem_data['time_step'] = omni_data['time_step']
    omni_dem_data['time'] = []
    omni_dem_data['position'] = []
    omni_dem_data['angle'] = []
    omni_dem_data['semantic_tag'] = []
    omni_dem_data['image_path'] = []

    quad_dem_data = {}
    quad_dem_data['scene_name'] = quad_data['scene_name']
    quad_dem_data['time_step'] = quad_data['time_step']
    quad_dem_data['time'] = []
    quad_dem_data['position'] = []
    quad_dem_data['angle'] = []
    quad_dem_data['semantic_tag'] = []
    quad_dem_data['image_path_quad'] = []

    semantic_tag_list = omni_data['semantic_tag']
    image_path_list = omni_data['image_path']
    image_path_quad_list = quad_data['image_path_quad']

    for index in range(len(quad_index_list)):
        if quad_index_list[index] != -1:
            # recordable
            position = position_list[index]
            angle = angle_list[index]
            time = -1
            image_path = image_path_list[index]
            image_path_quad = image_path_quad_list[index]
            semantic_tag = semantic_tag_list[index]
            omni_dem_data['time'].append(time)
            omni_dem_data['position'].append(position)
            omni_dem_data['angle'].append(angle)
            omni_dem_data['semantic_tag'].append(semantic_tag)
            omni_dem_data['image_path'].append(image_path)

            quad_dem_data['time'].append(time)
            quad_dem_data['position'].append(position)
            quad_dem_data['angle'].append(angle)
            quad_dem_data['semantic_tag'].append(semantic_tag)
            quad_dem_data['image_path_quad'].append(image_path_quad)

    with open(omni_location_bag_path, 'w') as f:
        json.dump(omni_dem_data, f)
    print("\nsuccessfully create the omni-directional location bag file {}\n".format(omni_location_bag_path))
    with open(quad_location_bag_path, 'w') as f:
        json.dump(quad_dem_data, f)
    print("\nsuccessfully create the quad-monocular location bag file {}\n".format(quad_location_bag_path))

