import os
import pdb
import json
import random
import numpy as np

"""
this file collect all the gather files in a root directory of a dataset
you can feed a txt file to specified the rooms you want to split out to validation set
otherwise, it will randomly choose 1/5 rooms to split out and save the names to a txt for future usage
this file is same as "CG_trainx_location.txt" in previous version of dataset
it saves the following information:
    image_id: int; ID of image
    node_id: string; ID of node
    room_id: string; ID of room
    pos_x: float
    pos_y: float
    pos_z: float
    ang_x: float
    ang_y: float
    ang_z: float
    image_subpath: string; partial path of the image (from root directory)
"""

if __name__ == '__main__':
    root_dir = raw_input('>>> Final test root directory (enter "d" to use default: /home/joehuang/Desktop/room_data_node_omni): ')
    if root_dir == 'd':
        root_dir = '/home/joehuang/Desktop/room_data_node_omni'
    gather_path_list = []
    for filename in os.listdir(root_dir):
        if 'gather' in filename:
            gather_path = os.path.join(root_dir, filename)
            print("data gather file {} collected".format(gather_path))
            gather_path_list.append(gather_path)


    room_id_unique_list = [] # all the unique room_id
    node_id_list = []
    room_id_list = []
    position_list = []
    angle_list = []
    image_subpath_list =[]
    print("start reading the gather files")
    for gather_path in gather_path_list:
        with open(gather_path) as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                else:
                    image_info = line[:-1].split(" ")
                    node_id = image_info[1]
                    room_id = image_info[2]
                    position = [image_info[3], image_info[4], image_info[5]]
                    angle = [image_info[6], image_info[7], image_info[8]]
                    image_path = image_info[9]
                    if image_path[:len(root_dir)] != root_dir:
                        raise ValueError("root directory does not match the file saved in the gather file")
                    image_subpath = image_path[len(root_dir)+1:]
                    node_id_list.append(node_id)
                    room_id_list.append(room_id)
                    position_list.append(position)
                    angle_list.append(angle)
                    image_subpath_list.append(image_subpath)
                    if room_id not in room_id_unique_list:
                        room_id_unique_list.append(room_id)
    print("done reading the gather files")

    split_filename = raw_input('>>> please input the filename that record the split strategy, if cannot find the path, the program will randomly split and save it in the specified path (enter "d" to use default split1.txt ): ')
    if split_filename == 'd':
        split_filename = 'split1.txt'
    suffix = raw_input('>>> suffix of saved txt file, ex: typing "1" will make train txt "CG_train1.txt": ')

    split_path = os.path.join(root_dir, split_filename)
    if not os.path.exists(split_path):
        # randomly split
        test_room_count = len(room_id_unique_list) / 5
        test_room_id_list = random.sample(room_id_unique_list, test_room_count)
        with open(split_path, 'w') as f:
            for test_room_id in test_room_id_list:
                f.write("{}\n".format(test_room_id))
    else:
        test_room_id_list = []
        with open(split_path, 'r') as f:
            for line in f:
                test_room_id_list.append(line[:-1])

    for test_room_id in test_room_id_list:
        print("{} to test file".format(test_room_id))

    test_image_id = 0
    train_image_id = 0
    train_path = os.path.join(root_dir, "CG_train" + suffix +".txt")
    test_path = os.path.join(root_dir, "CG_test" + suffix +".txt")
    f_train = open(train_path, 'w')
    f_test = open(test_path, 'w')
    f_train.write("image_id node_id room_id pos_x pos_y pos_z ang_x ang_y ang_z image_subpath\n")
    f_test.write("image_id node_id room_id pos_x pos_y pos_z ang_x ang_y ang_z image_subpath\n")
    for index in range(len(node_id_list)):
        node_id = node_id_list[index]
        room_id = room_id_list[index]
        position = position_list[index]
        angle = angle_list[index]
        image_subpath = image_subpath_list[index]
        if room_id in test_room_id_list:
            # this belongs to test
            f_test.write('{} {} {} {} {} {} {} {} {} {}\n'.format(test_image_id, node_id, room_id, position[0], position[1], position[2],
                    angle[0], angle[1], angle[2], image_subpath))
            test_image_id += 1
        else:
            # this belongs to train
            f_train.write('{} {} {} {} {} {} {} {} {} {}\n'.format(train_image_id, node_id, room_id, position[0], position[1], position[2],
                    angle[0], angle[1], angle[2], image_subpath))
            train_image_id += 1
    f_train.close()
    f_test.close()
    print("successfully split to train and test files")

