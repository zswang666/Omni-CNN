import os
import sys
import numpy as np
import pdb
import cv2
import itertools
import math
import random
import argparse

"""
reading training data without using a queue
will recreate batch while the batch list has been gone through
"""

def load_and_process_batch(path_batch, label_batch, rotation_batch, position_batch):
    """
    process the batch to real numpy batch
    Input:
        path_batch: list of string; list of file path/name
        label_batch: list of int; list of label
        rotation_batch: list of float; list of rotation angles
        position_batch: list of np.1darray; list of [x,z]
    Return:
        IMPORTANT!!: first B/2 are same tag, later B/2 are different tag from first B/2
        images: np.4darray; images batch
        labels: np.1darray; labels batch (B)
        rotation: np.1darray; rotation batch (B)
        position: np.2darray; position batch (B)
    """
    images = []
    labels = []
    rotations = []
    positions = []
    for index in range(len(path_batch)):
        image_path =path_batch[index]
        label = label_batch[index]
        rotation = rotation_batch[index]
        position = position_batch[index]
        image_data = cv2.imread(image_path)
        image_resize = cv2.resize(image_data, (0,0), fx=0.5, fy=0.5)

        images.append(image_resize)
        labels.append(label)
        rotations.append(rotation)
        positions.append(position)

    return np.array(images), np.array(labels), np.array(rotations), np.array(positions)

def create_batch_list(data_filename, data_dir, batch_size, total_batch_number):
    """
    read the data collection txt file and create batch list
    Input:
        data_filename: string; the data collection txt filename
        data_dir: string; the dataset directory
        batch_size: int; the batch size
        total_batch_number: int; the batch number that this file will generate
    Return:
        path_batch_list: list of list of string; list of batch (list of full path)
        label_batch_list: list of list of int; list of batch (list of labels)
        rotation_batch_list: list of list of float; list of batch (list of rotation angles)
        position_batch_list: list of list of np.1darray (x,z); list of batch (list of positions (x.z))
    """
    data_path = os.path.join(os.path.abspath(data_dir), os.path.relpath(data_filename))
    print("creating data batch using {} collection file".format(data_path))
    if not os.path.exists(data_path):
        print("data path {} not exists".format(data_path))
        sys.exit()

    # read the txt file, collect image path and its tags with rotation and position
    room_dict = {}
    inv_room_dict = {}
    image_dict = {}
    image_id_list = []
    num_images = 0
    with open(data_path) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            else:
                [image_id, room_id, pos_x, pos_y, pos_z, ang_x, ang_y, ang_z, image_path] = line[:-1].split(" ")
                num_images += 1
                image_id = int(image_id)
                ang_y = float(ang_y)
                pos_x = float(pos_x)
                pos_z = float(pos_z)
                image_id_list.append(image_id)
                image_path = os.path.join(os.path.abspath(data_dir), os.path.relpath(image_path))
                image_dict[image_id] = (image_path, ang_y, [pos_x, pos_z])
                inv_room_dict[image_id] = room_id
                if room_id not in room_dict:
                    room_dict[room_id] = [image_id]
                else:
                    room_dict[room_id].append(image_id)

    for room_id in room_dict.keys():
        # filter out rooms that have too less images to even create a batch
        if len(room_dict[room_id]) < batch_size / 2:
            image_ids = room_dict[room_id]
            for image_id in image_ids:
                image_id_list.remove(image_id)
                del inv_room_dict[image_id]
                del image_dict[image_id]
            del room_dict[room_id]

    room_list = room_dict.keys()
    room_image_count = [len(room_dict[room_id]) for room_id in room_list]
    room_list_probability = np.array(room_image_count, dtype=np.float32) / sum(room_image_count)
    room_choice = np.random.choice(room_list, total_batch_number, p=room_list_probability)
    unique, counts = np.unique(room_choice, return_counts=True)
    room_spreadout = dict(zip(unique, counts)) # room_id to batches targeted on that room

    path_batch_list = []
    label_batch_list = []
    rotation_batch_list = []
    position_batch_list = []
    for room_id in room_spreadout.keys():
        batch_count = room_spreadout[room_id]
        positive_image_ids = room_dict[room_id]
        groundset = set(image_id_list)
        negative_image_ids = groundset.difference(set(positive_image_ids))
        for _ in range(batch_count):
            path_batch = []
            label_batch = []
            rotation_batch = []
            position_batch = []

            positive_image_batch_ids = random.sample(positive_image_ids, batch_size/2)
            negative_image_batch_ids = random.sample(negative_image_ids, batch_size/2)
            for image_id in positive_image_batch_ids:
                image_path, image_rotation, image_position = image_dict[image_id]
                image_label = inv_room_dict[image_id]
                path_batch.append(image_path)
                label_batch.append(image_label)
                rotation_batch.append(image_rotation)
                position_batch.append(image_position)

            for image_id in negative_image_batch_ids:
                image_path, image_rotation, image_position = image_dict[image_id]
                image_label = inv_room_dict[image_id]
                path_batch.append(image_path)
                label_batch.append(image_label)
                rotation_batch.append(image_rotation)
                position_batch.append(image_position)

            path_batch_list.append(path_batch)
            label_batch_list.append(label_batch)
            rotation_batch_list.append(rotation_batch)
            position_batch_list.append(position_batch)

    # shuffle batches
    perm = range(len(path_batch_list))
    random.shuffle(perm)
    path_batch_list = [path_batch_list[index] for index in perm]
    label_batch_list = [label_batch_list[index] for index in perm]
    rotation_batch_list = [rotation_batch_list[index] for index in perm]
    position_batch_list = [position_batch_list[index] for index in perm]

    return path_batch_list, label_batch_list, rotation_batch_list, position_batch_list

class Reader:
    """
    The data loader for distance metric learning (continuous metric learning)
    Input:
        FLAGS: parameters given from user defined input
            data_filename: string; the filename of the data collection txt file
            data_dir: string; the directory of the whole data set
            batch_size: int; the batch size
            total_batch_number: int; the total number of batches to be generated
    """
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.data_filename = FLAGS.data_filename
        self.data_dir = FLAGS.data_dir
        self.batch_size = FLAGS.batch_size
        self.total_batch_number = FLAGS.total_batch_number

        self.path_batch_list, self.label_batch_list, self.rotation_batch_list, self.position_batch_list = create_batch_list(self.data_filename, self.data_dir, self.batch_size, self.total_batch_number)

        self.batch_num = 0

    def next_batch(self):
        """
        Return next batch, recreate batch list while the list is empty!
        Input:
            None
        Return:
            images: np.4darray; images batch
            labels: np.1darray; room tag batch
            rotations: np.1darray; rotation degree batch
            positions: np.2darray; (B,2) position x,z batch
        """
        if self.batch_num == self.total_batch_number:
            print("\n recreating batch list!! This might take few minutes, please wait\n")
            self.path_batch_list, self.label_batch_list, self.rotation_batch_list, self.position_batch_list = create_batch_list(self.data_filename, self.data_dir, self.batch_size, self.total_batch_number)
            self.batch_num = 0

        path_batch = self.path_batch_list[self.batch_num]
        label_batch = self.label_batch_list[self.batch_num]
        rotation_batch = self.rotation_batch_list[self.batch_num]
        position_batch = self.position_batch_list[self.batch_num]
        images, labels, rotations, positions = load_and_process_batch(path_batch, label_batch, rotation_batch, position_batch)
        self.batch_num += 1
        return images, labels, rotations, positions

if __name__ == '__main__':
    print("this class only feeding data for training")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/joehuang/Desktop/room_data_continuous', help='data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--total_batch_number', type=int, default=1000, help='total batch_number')
    parser.add_argument('--data_filename', type=str, default='CG_train1_location.txt', help='filename containing data list')
    FLAGS, unparsed = parser.parse_known_args()
    reader = Reader(FLAGS)

    for _ in range(5):
        images, labels, rotations, positions = reader.next_batch()
        for i in range(32):
            image = images[i]
            cv2.imshow("show", image)
            cv2.waitKey(0)
            print "class %d"%(labels[i])
        pdb.set_trace()
    print "test passed!"
