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
import pdb
import real_sfm

"""
this function take in a root directory of fish eye raw image and packed them into demonstration dataset and evaluation dataset
it can transform both omni dataset and quad dataset
omni location bag (demonstration) dataset style: json
    scene_name: string
    time_step: float, will all be -1.0
    time: list of float, will all be -1.0
    position: list of list of float (x,y,z)
    angle: list of list of float (x,y,z)
    semantic_tag: list of string; the room tag
    image_path: list of string; list of image paths
quad location bag (demonstration) dataset style: json
    scene_name: string
    time_step: float, will all be -1.0
    time: list of float, will all be -1.0
    position: list of list of float (x,y,z)
    angle: list of list of float (x,y,z)
    semantic_tag: list of string; the room tag
    image_path_quad: list of list of string (length 4); list of image path quad
omni test (evaluation) dataset style: txt
    image_id: int; image unique id in this set
    pos_x: float
    pos_y: float
    pos_z: float
    ang_x: float
    ang_y: float
    ang_z: float
    semantic_tag: string; the room id of the equi image
    image_path: string; full path of the equi image
quad test (evaluation) dataset style: txt
    image_id: int; image unique id in this set
    pos_x: float
    pos_y: float
    pos_z: float
    ang_x: float
    ang_y: float
    ang_z: float
    semantic_tag: string; the room id of the quad images
    original_image_path: string; full path of the front image
    image_path_90: string; full path of right image
    image_path_180: string; full path of back image
    image_path_270: string; full path of left image
"""

if __name__ == '__main__':
    # ask for root directory
    root_dir = raw_input('>>> Final realworld data root directory (enter "d" to use default: /home/joehuang/Desktop/Final_Test_Real): ')
    if root_dir == 'd':
        root_dir = '/home/joehuang/Desktop/Final_Test_Real'
    demonstration_dir = os.path.join(root_dir, 'demonstration')
    evaluation_dir = os.path.join(root_dir, 'evaluation')

    # ask for scene name, ex: EECS, look for EECS_fish, raise if file not exists
    scene_name = raw_input('>>> please input the name of the scene, ex: MSHouse (enter "d" to use default: MSHouse): ')
    if scene_name == 'd':
        scene_name = 'MSHouse'
    fisheye_dem_dir = os.path.join(demonstration_dir, scene_name + '-fish')
    fisheye_eval_dir = os.path.join(evaluation_dir, scene_name + '-fish')
    mono_dem_dir = os.path.join(demonstration_dir, scene_name + '-mono')
    mono_eval_dir = os.path.join(evaluation_dir, scene_name + '-mono')
    if not os.path.exists(fisheye_dem_dir):
        print("real world fisheye data demonstration file {} not found".format(fisheye_dem_dir))
        print("Program Quit!")
        sys.exit()
    if not os.path.exists(fisheye_eval_dir):
        print("real world fisheye data evaluation file {} not found".format(fisheye_eval_dir))
        print("Program Quit!")
        sys.exit()
    if not os.path.exists(mono_dem_dir):
        print("real world monocular data demonstration file {} not found".format(mono_dem_dir))
        print("Program Quit!")
        sys.exit()
    if not os.path.exists(mono_eval_dir):
        print("real world monocular data evaluation file {} not found".format(mono_eval_dir))
        print("Program Quit!")
        sys.exit()

    # check scene + '_converted.txt' exists or not, ex: EECS_converted.txt
    dem_omni_converted_collection_filename = scene_name + '_converted_dem_omni.json'
    dem_omni_converted_collection_path = os.path.join(root_dir, dem_omni_converted_collection_filename)
    dem_quad_converted_collection_filename = scene_name + '_converted_dem_quad.json'
    dem_quad_converted_collection_path = os.path.join(root_dir, dem_quad_converted_collection_filename)

    eval_omni_converted_collection_filename = scene_name + '_converted_eval_omni.json'
    eval_omni_converted_collection_path = os.path.join(root_dir, eval_omni_converted_collection_filename)
    eval_quad_converted_collection_filename = scene_name + '_converted_eval_quad.json'
    eval_quad_converted_collection_path = os.path.join(root_dir, eval_quad_converted_collection_filename)

    omni_location_bag_filename = scene_name + '_dem_omni.json'
    omni_location_bag_path = os.path.join(root_dir, omni_location_bag_filename)
    quad_location_bag_filename = scene_name + '_dem_quad.json'
    quad_location_bag_path = os.path.join(root_dir, quad_location_bag_filename)

    omni_test_filename = scene_name + '_eval_omni.txt'
    omni_test_path = os.path.join(root_dir, omni_test_filename)
    quad_test_filename = scene_name + '_eval_quad.txt'
    quad_test_path = os.path.join(root_dir, quad_test_filename)

    if os.path.exists(omni_location_bag_path):
        print("the location bag {} and {} already exists! Please delete it if you want to recreate the location bag file".format(omni_location_bag_path, quad_location_bag_path))
        sys.exit()

    if os.path.exists(omni_test_path):
        print("the test bag {} and {} already exists! Please delete it if you want to recreate the test bag file".format(omni_test_path, quad_test_path))
        sys.exit()

    dem_convert_fisheye_images_flag = True
    if os.path.exists(dem_omni_converted_collection_path):
        print("converted collection file is found")
        dem_convert_fisheye_images = raw_input('>>> do you want to re-convert demonstration fisheye images to equi-rectangular and cubic images? (y/n)')
        if dem_convert_fisheye_images == 'y':
            print("you should delete original transformed demonstration images and convert collection files then run this file again")
            print("Program Quit!")
            sys.exit()
        elif dem_convert_fisheye_images == 'n':
            dem_convert_fisheye_images_flag = False
        else:
            raise ValueError("unknown flag, please answer n or y")


    if dem_convert_fisheye_images_flag:
        # this is for converting demonstration fisheye images
        print("convert demonstration fisheye images to equirectangular images and quadrupal monocular images")
        dem_omni_data = {}
        dem_omni_data['scene_name'] = scene_name
        dem_omni_data['position'] = []
        dem_omni_data['angle'] = []
        dem_omni_data['semantic_tag'] = []
        dem_omni_data['image_path'] = []

        dem_quad_data = {}
        dem_quad_data['scene_name'] = scene_name
        dem_quad_data['position'] = []
        dem_quad_data['angle'] = []
        dem_quad_data['semantic_tag'] = []
        dem_quad_data['image_path_quad'] = []

        print("collect all demonstration fisheye image paths, and define all quad image paths, and equi image paths")
        quad_subdir = scene_name + '-quad'
        quad_dir = os.path.join(demonstration_dir, quad_subdir)
        omni_subdir = scene_name + '-omni'
        omni_dir = os.path.join(demonstration_dir, omni_subdir)
        print("create both demonstration omni-directional image directory {} and quad-monocular image directory {}".format(omni_dir, quad_dir))
        os.mkdir(omni_dir)
        os.mkdir(quad_dir)
        image_count = 0
        for room_subdir in os.listdir(fisheye_dem_dir):
            room_id = room_subdir #also semantic tag
            room_dir = os.path.join(fisheye_dem_dir, room_subdir)
            for image_filename in os.listdir(room_dir):
                fish_image_path = os.path.join(room_dir, image_filename)
                equi_image_filename = image_filename
                front_image_filename = image_filename[:-4] + '_1.jpg'
                right_image_filename = image_filename[:-4] + '_2.jpg'
                back_image_filename = image_filename[:-4] + '_3.jpg'
                left_image_filename = image_filename[:-4] + '_4.jpg'

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

                dem_omni_data['semantic_tag'].append(room_id)
                dem_omni_data['image_path'].append(equi_image_path)

                dem_quad_data['semantic_tag'].append(room_id)
                dem_quad_data['image_path_quad'].append([front_image_path, right_image_path, back_image_path, left_image_path])
                image_count += 1
                print("successfully processed {} demonstration images".format(image_count))
        with open(dem_omni_converted_collection_path, 'w') as f:
            json.dump(dem_omni_data, f)
        print("successfully saved omni converted collection file in {}".format(dem_omni_converted_collection_path))
        with open(dem_quad_converted_collection_path, 'w') as f:
            json.dump(dem_quad_data, f)
        print("successfully saved quad converted collection file in {}".format(dem_quad_converted_collection_path))
    else:
        print("use old equirectangular images and quadrupal monocular images")
        with open(dem_omni_converted_collection_path) as f:
            dem_omni_data = json.load(f)
        with open(dem_quad_converted_collection_path) as f:
            dem_quad_data = json.load(f)

    eval_convert_fisheye_images_flag = True
    if os.path.exists(eval_omni_converted_collection_path):
        print("converted collection file is found")
        eval_convert_fisheye_images = raw_input('>>> do you want to re-convert evaluation fisheye images to equi-rectangular and cubic images? (y/n)')
        if eval_convert_fisheye_images == 'y':
            print("you should delete original transformed evluation images and convert collection files then run this file again")
            print("Program Quit!")
            sys.exit()
        elif eval_convert_fisheye_images == 'n':
            eval_convert_fisheye_images_flag = False
        else:
            raise ValueError("unknown flag, please answer n or y")


    if eval_convert_fisheye_images_flag:
        # this is for converting demonstration fisheye images
        print("convert evaluation fisheye images to equirectangular images and quadrupal monocular images")
        eval_omni_data = {}
        eval_omni_data['scene_name'] = scene_name
        eval_omni_data['position'] = []
        eval_omni_data['angle'] = []
        eval_omni_data['semantic_tag'] = []
        eval_omni_data['image_path'] = []

        eval_quad_data = {}
        eval_quad_data['scene_name'] = scene_name
        eval_quad_data['position'] = []
        eval_quad_data['angle'] = []
        eval_quad_data['semantic_tag'] = []
        eval_quad_data['image_path_quad'] = []

        print("collect all evaluation fisheye image paths, and define all quad image paths, and equi image paths")
        quad_subdir = scene_name + '-quad'
        quad_dir = os.path.join(evaluation_dir, quad_subdir)
        omni_subdir = scene_name + '-omni'
        omni_dir = os.path.join(evaluation_dir, omni_subdir)
        print("create both evaluation omni-directional image directory {} and quad-monocular image directory {}".format(omni_dir, quad_dir))
        os.mkdir(omni_dir)
        os.mkdir(quad_dir)
        image_count = 0
        for room_subdir in os.listdir(fisheye_eval_dir):
            room_id = room_subdir #also semantic tag
            room_dir = os.path.join(fisheye_eval_dir, room_subdir)
            for image_filename in os.listdir(room_dir):
                fish_image_path = os.path.join(room_dir, image_filename)
                equi_image_filename = image_filename
                front_image_filename = image_filename[:-4] + '_1.jpg'
                right_image_filename = image_filename[:-4] + '_2.jpg'
                back_image_filename = image_filename[:-4] + '_3.jpg'
                left_image_filename = image_filename[:-4] + '_4.jpg'

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

                eval_omni_data['semantic_tag'].append(room_id)
                eval_omni_data['image_path'].append(equi_image_path)

                eval_quad_data['semantic_tag'].append(room_id)
                eval_quad_data['image_path_quad'].append([front_image_path, right_image_path, back_image_path, left_image_path])
                image_count += 1
                print("successfully processed {} evaluation images".format(image_count))
        with open(eval_omni_converted_collection_path, 'w') as f:
            json.dump(eval_omni_data, f)
        print("successfully saved omni converted collection file in {}".format(eval_omni_converted_collection_path))
        with open(eval_quad_converted_collection_path, 'w') as f:
            json.dump(eval_quad_data, f)
        print("successfully saved quad converted collection file in {}".format(eval_quad_converted_collection_path))
    else:
        print("use old equirectangular images and quadrupal monocular images")
        with open(eval_omni_converted_collection_path) as f:
            eval_omni_data = json.load(f)
        with open(eval_quad_converted_collection_path) as f:
            eval_quad_data = json.load(f)

    # get data file path in monocular data
    mono_image_filename_list = []
    for room_subdir in os.listdir(mono_eval_dir):
        room_id = room_subdir #also semantic tag
        room_dir = os.path.join(mono_eval_dir, room_subdir)
        for image_filename in os.listdir(room_dir):
            mono_image_filename_list.append(image_filename)
    for room_subdir in os.listdir(mono_dem_dir):
        room_id = room_subdir #also semantic tag
        room_dir = os.path.join(mono_dem_dir, room_subdir)
        for image_filename in os.listdir(room_dir):
            mono_image_filename_list.append(image_filename)


    print("\nstart running opensfm to get demonstration image locations\n")
    filename2location = real_sfm.run_sfm(scene_name, mono_image_filename_list)
    print("\nsuccessfully running sfm and convert into {}/{} positions/paths\n".format(len(filename2location), len(mono_image_filename_list)))
    for image_filename in filename2location.keys():
        print image_filename
        print filename2location[image_filename]
    succeed_filename = filename2location.keys()
    # done with sfm


    # gather and save dem data for quad and omni
    sfm_dem_omni_data = {}
    sfm_dem_omni_data['scene_name'] = dem_omni_data['scene_name']
    sfm_dem_omni_data['time_step'] = -1.
    sfm_dem_omni_data['time'] = []
    sfm_dem_omni_data['position'] = []
    sfm_dem_omni_data['angle'] = []
    sfm_dem_omni_data['semantic_tag'] = []
    sfm_dem_omni_data['image_path'] = []

    sfm_dem_quad_data = {}
    sfm_dem_quad_data['scene_name'] = dem_quad_data['scene_name']
    sfm_dem_quad_data['time_step'] = -1.
    sfm_dem_quad_data['time'] = []
    sfm_dem_quad_data['position'] = []
    sfm_dem_quad_data['angle'] = []
    sfm_dem_quad_data['semantic_tag'] = []
    sfm_dem_quad_data['image_path_quad'] = []

    dem_semantic_tag_list = dem_omni_data['semantic_tag']
    dem_image_path_list = dem_omni_data['image_path']
    dem_image_path_quad_list = dem_quad_data['image_path_quad']

    for index in range(len(dem_image_path_list)):
        dem_image_path = dem_image_path_list[index]
        dem_image_filename = os.path.basename(dem_image_path)
        if dem_image_filename in succeed_filename:
            location = filename2location[dem_image_filename]
            position = list(location[0])
            angle = list(location[1])
            time = -1
            image_path = dem_image_path_list[index]
            image_path_quad = dem_image_path_quad_list[index]
            semantic_tag = dem_semantic_tag_list[index]
            sfm_dem_omni_data['time'].append(time)
            sfm_dem_omni_data['position'].append(position)
            sfm_dem_omni_data['angle'].append(angle)
            sfm_dem_omni_data['semantic_tag'].append(semantic_tag)
            sfm_dem_omni_data['image_path'].append(image_path)

            sfm_dem_quad_data['time'].append(time)
            sfm_dem_quad_data['position'].append(position)
            sfm_dem_quad_data['angle'].append(angle)
            sfm_dem_quad_data['semantic_tag'].append(semantic_tag)
            sfm_dem_quad_data['image_path_quad'].append(image_path_quad)

    with open(omni_location_bag_path, 'w') as f:
        json.dump(sfm_dem_omni_data, f)
    print("\nsuccessfully create the demonstration omni-directional location bag file {}\n".format(omni_location_bag_path))
    with open(quad_location_bag_path, 'w') as f:
        json.dump(sfm_dem_quad_data, f)
    print("\nsuccessfully create the demonstration quad-monocular location bag file {}\n".format(quad_location_bag_path))


    # gather and save eval data for quad and omni
    sfm_eval_omni_data_list = []
    sfm_eval_quad_data_list = []

    eval_semantic_tag_list = eval_omni_data['semantic_tag']
    eval_image_path_list = eval_omni_data['image_path']
    eval_image_path_quad_list = eval_quad_data['image_path_quad']

    image_count = 0
    for index in range(len(eval_image_path_list)):
        eval_image_path = eval_image_path_list[index]
        eval_image_filename = os.path.basename(eval_image_path)
        if eval_image_filename in succeed_filename:
            location = filename2location[eval_image_filename]
            position = list(location[0])
            angle = list(location[1])
            image_path = eval_image_path_list[index]
            image_path_quad = eval_image_path_quad_list[index]
            semantic_tag = eval_semantic_tag_list[index]
            omni_image_data = "{} {} {} {} {} {} {} {} {}\n".format(str(image_count), position[0], position[1], position[2], angle[0], angle[1], angle[2], semantic_tag, image_path)
            sfm_eval_omni_data_list.append(omni_image_data)

            quad_image_data = "{} {} {} {} {} {} {} {} {} {} {} {}\n".format(str(image_count), position[0], position[1], position[2], angle[0], angle[1], angle[2],
                    semantic_tag, image_path_quad[0], image_path_quad[1], image_path_quad[2], image_path_quad[3])
            sfm_eval_quad_data_list.append(quad_image_data)
            image_count += 1

    with open(omni_test_path, 'w') as f:
        f.write("image_id pos_x pos_y pos_z ang_x ang_y ang_z semantic_tag original_image_path\n")
        for sfm_eval_omni_data in sfm_eval_omni_data_list:
            f.write(sfm_eval_omni_data)
    print("\nsuccessfully create evaluation omni-directional test file {}\n".format(omni_test_path))

    with open(quad_test_path, 'w') as f:
        f.write("image_id pos_x pos_y pos_z ang_x ang_y ang_z semantic_tag original_image_path image_path_90 image_path_180 image_path_270\n")
        for sfm_eval_quad_data in sfm_eval_quad_data_list:
            f.write(sfm_eval_quad_data)
    print("\nsuccessfully create evaluation quad-mono test file {}\n".format(quad_test_path))
