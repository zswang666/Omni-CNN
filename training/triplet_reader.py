import sys
sys.path.append("../")
import argparse
import importlib
import os
import cv2
import random
import pdb
import itertools
import numpy as np
import copy

class Triplet_Reader:
    def __init__(self, FLAGS, control_distances):
        """
        the estimation class to estimate the accuracy of the current model
        this class is used in dist version, can choose to give same room or different
        you may control the distance from the positive and the tolerance margin
        Input:
            FLAGS: flags for this metric
                data_filename: string; the data collection txt filename for evaluation
                data_dir: string; the data directory that stores all images
                batch_size: int; the size of a batch
                tolerance_margin: float; the negative should have distance larger than margin + pos dist
                pick_same_room: bool; pick the same room
            control_distances: list of tuple of float; the positive pair should have distance less than this
                ex: [(0.,0.5), (0.5,1.0), (1.0,1.5), (1.5,2.0), (2.0, 2.5)] should be good
        """
        self.data_filename = FLAGS.data_filename
        self.data_dir = FLAGS.data_dir
        self.batch_size = FLAGS.batch_size
        self.control_distances = control_distances
        self.tolerance_margin = FLAGS.tolerance_margin
        self.pick_same_room = FLAGS.pick_same_room

        pick_same_room = True
        data_path = os.path.join(os.path.abspath(self.data_dir), self.data_filename)
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
                    image_path = os.path.join(os.path.abspath(self.data_dir), os.path.relpath(image_path))
                    image_dict[image_id] = (image_path, ang_y, [pos_x, pos_z])
                    inv_room_dict[image_id] = room_id
                    if room_id not in room_dict:
                        room_dict[room_id] = [image_id]
                    else:
                        room_dict[room_id].append(image_id)

        room_list = room_dict.keys()
        room_image_count = [len(room_dict[room_id]) for room_id in room_list]
        room_list_probability = np.array(room_image_count, dtype=np.float32) / sum(room_image_count)

        self.control_distance_batch_list = [] # list of batch list based on different distance control
        for control_distance in self.control_distances:
            anc_pos_neg_rot_batch_list = []
            min_distance = control_distance[0]
            max_distance = control_distance[1]
            for _ in range(20):
                anc_pos_neg_rot_batch = []
                while len(anc_pos_neg_rot_batch) != self.batch_size:
                    # try five times, if can't then random choose another room_id
                    room_id = np.random.choice(room_list, p=room_list_probability)
                    positive_image_ids = copy.deepcopy(room_dict[room_id])
                    random.shuffle(positive_image_ids)
                    # random pick an id as anchor id
                    anchor_id = random.choice(positive_image_ids)
                    anchor_pos = np.array(image_dict[anchor_id][2])
                    # go through all other ids to find legal positive id
                    for positive_id in positive_image_ids:
                        positive_pos = np.array(image_dict[positive_id][2])
                        positive_distance = np.sqrt(np.sum(np.square(positive_pos - anchor_pos)))
                        if positive_id != anchor_id and positive_distance >= min_distance and positive_distance <= max_distance:
                            # legal positive pair, will end up adding a legal batch
                            if self.pick_same_room:
                                random.shuffle(positive_image_ids)
                                legal_negative_found = False
                                # find legal negative id in the same room
                                for negative_id in positive_image_ids:
                                    negative_pos = np.array(image_dict[negative_id][2])
                                    negative_distance = np.sqrt(np.sum(np.square(negative_pos - anchor_pos)))
                                    if negative_id != positive_id and negative_id != anchor_id:
                                        if negative_distance - positive_distance >= self.tolerance_margin:
                                            # negative id found, now create the batch
                                            legal_negative_found = True
                                            anchor_path = image_dict[anchor_id][0]
                                            positive_path = image_dict[positive_id][0]
                                            negative_path = image_dict[negative_id][0]
                                            rotation = image_dict[positive_id][1] - image_dict[anchor_id][1] # pos - anc
                                            anc_pos_neg_rot_batch.append((anchor_path, positive_path, negative_path, rotation))
                                            break
                                if not legal_negative_found:
                                    # pick another room any point as negative id
                                    groundset = set(image_id_list)
                                    negative_sample_set = groundset.difference(set(positive_image_ids))
                                    negative_id = random.choice(list(negative_sample_set))

                                    anchor_path = image_dict[anchor_id][0]
                                    positive_path = image_dict[positive_id][0]
                                    negative_path = image_dict[negative_id][0]
                                    rotation = image_dict[positive_id][1] - image_dict[anchor_id][1] # pos - anc
                                    anc_pos_neg_rot_batch.append((anchor_path, positive_path, negative_path, rotation))
                            else:
                                # do not have to pick the same room
                                groundset = set(image_id_list)
                                negative_sample_set = groundset.difference(set(positive_image_ids))
                                negative_id = random.choice(negative_sample_set)

                                anchor_path = image_dict[anchor_id][0]
                                positive_path = image_dict[positive_id][0]
                                negative_path = image_dict[negative_id][0]
                                rotation = image_dict[positive_id][1] - image_dict[anchor_id][1] # pos - anc
                                anc_pos_neg_rot_batch.append((anchor_path, positive_path, negative_path, rotation))
                            break
                random.shuffle(anc_pos_neg_rot_batch)
                anc_pos_neg_rot_batch_list.append(anc_pos_neg_rot_batch)
            self.control_distance_batch_list.append(anc_pos_neg_rot_batch_list)
        self.control_distance_batch_index = [0 for _ in range(len(self.control_distance_batch_list))]
        print("complete creating test batches")

    def process_batch(self, image_batch):
        """
        process the batch to real numpy batch
        Input:
            image_batch: list of cv2.images; batch of images
        Return:
            images: np.4darray; images batch
        """
        images = []
        for image_data in image_batch:
            image_resize = cv2.resize(image_data, (0,0), fx=0.5, fy=0.5)

            images.append(image_resize)
        return np.array(images)

    def next_batch(self, control_distance_index):
        """
        create evaluate batches by choosing which control distance batch you want
        Input:
            control_distance_index: int; the control distance will be self.control_distances[control_distance_index]
        Return:
            anchor_image_batch: np.4darray; anchor image batch
            positive_image_batch: np.4darray; positive image batch
            negative_image_batch: np.4darray; negative image batch
            rotation_batch: np.1darray; positive left turn this much become anchor
        """
        batch_index = self.control_distance_batch_index[control_distance_index]
        if (batch_index >= len(self.control_distance_batch_list[control_distance_index])):
            self.control_distance_batch_index[control_distance_index] = 0
            batch_index = 0
        anc_pos_neg_rot_batch = self.control_distance_batch_list[control_distance_index][batch_index]
        anchor_image_list = []
        positive_image_list = []
        negative_image_list = []
        rotation_list = [] # pos angle - anc angle
        for anc_pos_neg_rot_tuple in anc_pos_neg_rot_batch:
            anchor_image = cv2.imread(anc_pos_neg_rot_tuple[0])
            positive_image = cv2.imread(anc_pos_neg_rot_tuple[1])
            negative_image = cv2.imread(anc_pos_neg_rot_tuple[2])
            anchor_image_list.append(anchor_image)
            positive_image_list.append(positive_image)
            negative_image_list.append(negative_image)
            rotation_list.append(anc_pos_neg_rot_tuple[3])
        anchor_image_batch = self.process_batch(anchor_image_list)
        positive_image_batch = self.process_batch(positive_image_list)
        negative_image_batch = self.process_batch(negative_image_list)
        rotation_batch = np.array(rotation_list)
        self.control_distance_batch_index[control_distance_index] += 1
        return anchor_image_batch, positive_image_batch, negative_image_batch, rotation_batch

    def evaluate_correct_number(self, anchor_feature, positive_feature, negative_feature, rotation_batch):
        """
        evaluate the correct number in the features
        Input:
            anchor_feature: list of np.3darray; the anchor image feature L of [B,W,C]
            positive_feature: list of np.3darray; the positive image feature L of [B,W,C]
            negative_feature: list of np.3darray; the negative image feature L of [B,W,C]
            rotation_batch: np.1darray; the rotation [B]
        Return:
            correct_number: int; correct number
        """
        L = len(anchor_feature)
        B, W, C = anchor_feature[0].shape
        correct_number = 0
        for index in range(B):
            f_anc = anchor_feature[0][index]
            D_pos_list = []
            D_neg_list = []
            for rot in range(L):
                f_pos_rot = positive_feature[rot][index]
                f_neg_rot = negative_feature[rot][index]
                D_pos = np.sqrt(np.sum(np.square(f_anc - f_pos_rot)))
                D_neg = np.sqrt(np.sum(np.square(f_anc - f_neg_rot)))
                D_pos_list.append(D_pos)
                D_neg_list.append(D_neg)
            if min(D_pos_list) < min(D_neg_list):
                correct_number += 1
        return correct_number

    def evaluate_rotation_error(self, anchor_feature, positive_feature, negative_feature, rotation_batch):
        """
        evaluate the correct number in the features
        Input:
            anchor_feature: list of np.3darray; the anchor image feature L of [B,W,C]
            positive_feature: list of np.3darray; the positive image feature L of [B,W,C]
            negative_feature: list of np.3darray; the negative image feature L of [B,W,C]
            rotation_batch: np.1darray; the rotation [B]
        Return:
            avg_error: float; average error
        """
        L = len(anchor_feature)
        B, W, C = anchor_feature[0].shape
        error_list = []
        for index in range(B):
            f_anc = anchor_feature[0][index]
            D_pos_list = []
            for rot in range(L):
                f_pos_rot = positive_feature[rot][index]
                D_pos = np.sqrt(np.sum(np.square(f_anc - f_pos_rot)))
                D_pos_list.append(D_pos)
            rot_array = np.arange(L) * 360. / L
            if index == 0:
                print "ground truth rotation: {}".format(360 - rotation_batch[0]%360.)
                np.set_printoptions(precision=3)
                show = np.hstack((np.reshape(np.array(D_pos_list), (-1,1)), np.reshape(rot_array, (-1,1))))
                print(np.array(show))
                print ""
            # calculate the rotation error and return it
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str,
        default='CG_test1_location.txt',
        help='filename of collection file that collects all test image information')
    parser.add_argument('--data_dir', type=str,
        default='/home/joehuang/Desktop/room_data_continuous',
        help='the directory that contains the whole dataset')
    parser.add_argument('--batch_size', type=int,
        default=32,
        help='the batch size')
    parser.add_argument('--tolerance_margin', type=float,
        default=1.0,
        help='the tolerance margin')
    parser.add_argument('--pick_same_room', type=bool,
        default=True,
        help='pick the same room or not')
    FLAGS, _ = parser.parse_known_args()
    control_distances = [(0,0.5), (0.5, 1.0), (1.0, 1.5), (1.5,2.0)]
    triplet_reader = Triplet_Reader(FLAGS, control_distances)

    cv2.namedWindow('anchor',cv2.WINDOW_NORMAL)
    cv2.namedWindow('positive',cv2.WINDOW_NORMAL)
    cv2.namedWindow('negative',cv2.WINDOW_NORMAL)
    for _ in range(100):
        print "yo"
        anchor_image_batch, positive_image_batch, negative_image_batch, rotation_batch = triplet_reader.next_batch(2)
        anchor_image = anchor_image_batch[0]
        positive_image = positive_image_batch[0]
        negative_image = negative_image_batch[0]
        cv2.imshow('anchor', anchor_image)
        cv2.imshow('positive', positive_image)
        cv2.imshow('negative', negative_image)
        cv2.waitKey(0)
