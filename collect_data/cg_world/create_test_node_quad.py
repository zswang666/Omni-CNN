import os
import sys
sys.path.append('./socketapi')
import GameController as gc
import imgdecode as imd
import json
import cv2
import random
import numpy as np

"""
This file collects images and save it as a collection file, this file is for node and quad monocular camera
each setting will have to run this file once to create its collection file, the collection file is different between setup(node, continuous) and sensors(omni,quad,mono)
saved sampling file name will be: 'Scene_0_1-node-quad.txt' and will locate in root directory
for each image, the following information will be presented in order:
    image_id: int; the id of image
    node_id: string; the id of node, same in same room in same scene but different setting
    room_id: string; the id of room, same in same room in same scene but different setting
    pos_x: float;
    pos_y: float;
    pos_z: float;
    ang_x: float;
    ang_y: float;
    ang_z: float;
    image_path: string; the partially path of the image under the root directory
for the whole setting, the file will make a directory for it
for each room_tag, the file will make a directory in the setting directory for it and save images in it
"""


if __name__ == '__main__':
    root_dir = raw_input('>>> Final test root directory (enter "d" to use default: /home/joehuang/Desktop/room_data_node_quad): ')
    if root_dir == 'd':
        root_dir = '/home/joehuang/Desktop/room_data_node_quad'

    collection_filename = raw_input('>>> the collection filename, which is the file that records by human driving robot to collect points: (enter "d" to use default: Scene_7-root.json) ')
    if collection_filename == 'd':
        collection_filename = 'Scene_7-root.json'
    collection_path = os.path.join(root_dir, collection_filename)

    with open(collection_path, 'r') as f:
        json_data = json.load(f)
    scene_name = json_data['scene_name']
    node_ids = json_data['node_ids']
    room_ids = json_data['room_ids']
    positions = json_data['positions']
    angles = json_data['angles']

    setting_id = int(raw_input('>>> please input the setting id you want to use for scene {}: '.format(scene_name)))
    sampling_count = int(raw_input('>>> how many points you want to sample in each collected point? (suggesting: 10): '))
    sigma_xy = float(raw_input('>>> what is the sampling range in unity unit you want to sample? (suggesting: 0.05): '))

    scene_setting_name = scene_name + '_' + str(setting_id)
    sampling_filename = scene_setting_name + '_gather-node-quad.txt'
    sampling_path = os.path.join(root_dir, sampling_filename)
    scene_setting_dir = os.path.join(root_dir, scene_setting_name)
    if os.path.exists(scene_setting_dir):
        print("warning! scene setting directory {} exists, please delete it to recollect image data".format(scene_setting_dir))
        sys.exit()
    else:
        print("create scene setting directory {}".format(scene_setting_dir))
        os.mkdir(scene_setting_dir)

    raw_input('>>> warning: you should be expecting to collect omni-directional image and using node sampling style in this file, press any key to continue')
    raw_input('>>> remember to open {} unity execution file, press any key to continue'.format(scene_setting_name))

    try:
        con = gc.Controller()
        con.connect()
        print("successfully connect to game controller")
    except:
        print('Error: Socket Connection failed')
        print('did you open your unity exe file?')
        sys.exit()

    print("start sampling points and record images")

    image_id_list = []
    node_id_list = []
    room_id_list = []
    position_list = [] #list of list
    angle_list = []
    image_path_list = []

    quad_rotate_list = [0., 90., 180., 270.]
    image_count = 0
    for sampling_index in range(len(node_ids)):
        room_id = room_ids[sampling_index]
        node_id = node_ids[sampling_index]
        position = positions[sampling_index]
        angle = angles[sampling_index]
        if room_id not in room_id_list:
            # create a directory for the room
            room_dir = os.path.join(scene_setting_dir, room_id)
            print("create directory {} for the room".format(room_dir))
            os.mkdir(room_dir)
        for i in range(sampling_count):
            # delete move too far, y changes and connection failed points
            x = random.uniform(position[0] - sigma_xy, position[0] + sigma_xy)
            z = random.uniform(position[2] - sigma_xy, position[2] + sigma_xy)
            theta = random.uniform(0., 360.)
            for quad_rotate in quad_rotate_list:
                try:
                    con.setPos(x, position[1], z)
                    con.setRot(angle[0], theta + quad_rotate, angle[2])
                    sample_position = con.getPos()
                    sample_angle = con.getRot()
                    shifted = np.sqrt(np.sum(np.square(np.array(sample_position) - np.array(position))))
                    if (position[1] - sample_position[1]) >= 0.001:
                        print("DROPPED: vertically shifted")
                    elif shifted >= sigma_xy * 1.5:
                        print("DROPPED: 2d shifted more than 1.5 * sigma_xy")
                    else:
                        image_count += 1
                        print("point collected, image count: {}".format(image_count))
                        image_filename = node_id + '_' + str(i) + '.jpg'
                        image_path = os.path.join(room_dir, image_filename)
                        image = con.getSpherical()
                        image = imd.decode(image)
                        cv2.imwrite(image_path, image)
                        image_id_list.append(image_count)
                        node_id_list.append(node_id)
                        room_id_list.append(room_id)
                        position_list.append(sample_position)
                        angle_list.append(sample_angle)
                        image_path_list.append(image_path)
                except:
                    print("DROPPED: controller error")
                    con.close()
                    try:
                        con = gc.Controller()
                        con.connect()
                        print("successfully connect to game controller")
                    except:
                        print('Error: Socket Connection failed')
                        print('did you open your unity exe file?')
                        sys.exit()
    print("finish collecting samples, totaly {} samples collected".format(image_count))

    with open(sampling_path, 'w') as f:
        print("\nstart saving sampling txt file")
        f.write("image_id node_id room_id pos_x pos_y pos_z ang_x ang_y ang_z image_path\n")
        for index in range(len(image_id_list)):
            position = position_list[index]
            angle = angle_list[index]
            node_id = node_id_list[index]
            room_id = room_id_list[index]
            image_id = image_id_list[index]
            image_path = image_path_list[index]
            f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(image_id, node_id, room_id, position[0], position[1], position[2],
                    angle[0], angle[1], angle[2], image_path))
        print("test txt file successfully saved in: {}".format(sampling_path))
