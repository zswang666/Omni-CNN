import numpy as np
import pdb
import os
import json
import cv2
import argparse
import sys
import time
import math
from model import CaffeNet
sys.path.append("../../")
from interface import Localization_Evaluation_Interface
import tensorflow as tf
from filter_by_feature_mapping import filter_by_feature_mapping

class Localization_Evaluation(Localization_Evaluation_Interface):
    """
    class for evaluation of our algorithm
    In demonstration stage:
        (1) pick nodes that are not too close together
    In prediction stage:
        (1) semantic filter: pick closest N nodes among all nodes, collect their semantic tags,
                count them and pick the tag that is higher than a bar and collect all nodes under these tags
        (2) distance filter 1: pick closest N nodes among the nodes chosen above
        (3) feature matching: pick top k nodes that features matches best to query image among nodes chosen above
        (4) distance filter 2: pick the closest node among the above nodes
    """
    def __init__(self, FLAGS):
        """
        Input:
            FLAGS: The flags for this localization algorithm to run well
            min_node_distance: float; when forming node in demonstration file, minimum distance to set another node

            feature_size: int; the size of feature
            conv_initialization: string; 'random', 'pretrain' or 'trained', which are the way convolution layers initialized
            pretrain_path: string; if 'pretrain' is chosen above, then the pretrain path is notified here
            model_path: string; the path of model
            model_checkpoint: string; the model checkpoint to be loaded
            random_initialize_fc_layer: bool; random initialize fc layer or not
        """
        self.min_node_distance = FLAGS.min_node_distance #0.7

        batch_size = 1
        conv_initialization = FLAGS.conv_initialization
        pretrain_path = FLAGS.pretrain_path
        model_path = FLAGS.model_path
        model_checkpoint = FLAGS.model_checkpoint
        random_initialize_fc_layer = FLAGS.random_initialize_fc_layer

        # imagenet mean
        #images_mean = tf.constant([104./127.5 - 1.0, 117./127.5 -1.0, 123./127.5 - 1.0])
        # place365 mean
        images_mean = tf.constant([104./127.5 - 1.0, 113./127.5 -1.0, 117./127.5 - 1.0])
        self.images_placeholder = tf.placeholder(tf.float32,
                                         shape=(batch_size,384,640,3))
        images = self.images_placeholder / 127.5 - 1.0 - images_mean

        print('Building graph...')
        with tf.device('/gpu:0'):
            self.net = CaffeNet({'data': images})
            # before dropout
            self.conv_features = self.net.get_output()
            avg_pool = tf.nn.avg_pool(self.conv_features, ksize=[1,12,1,1], # NOTE
                    strides=[1, 1, 1, 1], padding='VALID', name='pool5_7x7_s1')
            B,H,W,C = avg_pool.get_shape().as_list()
            self.metric_features = tf.reshape(avg_pool, [B,W*H,C])

            t_list = tf.split(self.metric_features, num_or_size_splits=W, axis=1)
            indices = np.arange(W)
            self.out_branches = []
            for _ in range(W):
                # form a branch
                br = [t_list[idx] for idx in indices]
                br = tf.concat(br, axis = 1)
                # append new branch into branch list
                self.out_branches.append(br)
                # update indices, this uses numpy
                indices = np.roll(indices, -1)
            self.rolling_features_list = self.out_branches

            print('Finish building graph')

        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if conv_initialization == 'random':
            print("random initialized conv layers, did not load pretrain model")
        elif conv_initialization == 'pretrain':
            if os.path.exists(pretrain_path):
                print("load pretrain model from {}".format(pretrain_path))
                self.net.load(pretrain_path, self.sess, True)
            else:
                raise ValueError("pretrain model not exist in {}".format(pretrain_path))
        elif conv_initialization == 'trained':
            if model_checkpoint == 'latest':
                print("load latest checkpoint from {}".format(model_path))
                saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
            else:
                checkpoint_path = os.path.join(model_path, model_checkpoint)
                print("load checkpoint from {}".format(checkpoint_path))
                saver.restore(self.sess, checkpoint_path)
        else:
            raise ValueError("unknown conv initialization style")


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
            image_resize = cv2.resize(image_data, (0,0), fx=0.5, fy=0.5) #NOTE
            images.append(image_resize)

        return np.array(images)

    def demonstrate(self, train_path):
        """
        User demonstrate once to the robot
        Input:
            train_path: string; the demonstration information file path
        """
        if not os.path.exists(train_path):
            print("training json file not exists, program quit")
            sys.exit()
        with open(train_path) as f:
            json_data = json.load(f)
        self.train_time_stamp_list = json_data['time']
        self.train_image_path_list = json_data['image_path']
        self.train_position_list = json_data['position']
        self.train_angle_list = json_data['angle']
        self.train_semantic_tag_list = json_data['semantic_tag']
        num_images = len(self.train_image_path_list)

        # create nodes
        print("start demonstrating, totally {} images in demonstration set".format(num_images))
        self.node_id_list = []
        self.node_semantic_tag_list = []
        self.node_metric_feature_list = []
        self.node_conv_feature_list = []
        last_node_position = np.array([float('inf'), float('inf'), float('inf')])
        for train_index in range(num_images):
            train_position = np.array(self.train_position_list[train_index])
            if np.sqrt(np.sum(np.square(train_position - last_node_position))) > self.min_node_distance:
                last_node_position = train_position
                self.node_id_list.append(train_index)
                train_semantic_tag = self.train_semantic_tag_list[train_index]
                self.node_semantic_tag_list.append(train_semantic_tag)
                node_image_path = self.train_image_path_list[train_index]
                node_image = cv2.imread(node_image_path)
                image_batch = self.process_batch([node_image])
                node_conv_feature, node_metric_feature = self.sess.run([self.conv_features,
                        self.metric_features], feed_dict = {self.images_placeholder: image_batch})
                self.node_conv_feature_list.append(node_conv_feature[0])
                self.node_metric_feature_list.append(node_metric_feature[0])
            print("{}/{} demonstration image shown".format(train_index+1, num_images))
        self.node_number = len(self.node_id_list)
        print("all nodes created, totally {} of nodes".format(len(self.node_id_list)))

    def prediction(self, query_image_path_list, evaluation_method, demonstration_subdir, evaluation_subdir):
        """
        predict the location of the list of the images
        Input:
            query_image_path_list: list of string; the list of query image paths
            evaluation_method: sting; should be 'find_location'
            demonstration_subdir: the demonstration directory left in this localization method directory
            evaluation_subdir: unimportant in this algorithm
        Return:
            change due to different evaluation method:
                find_target_position:
                    es_semantic_tag_list: list of set; list of semantic tag set,ex:[set(room1, room2), set(room1)]
                    es_target_index_list: list of int; list of target index, which is their train index, node id
                target_position_precision-error:
                    es_target_index_list: list of int; list of target index, which is their train index, node id
                target_position_distance-accuracy:
                    es_target_index_list: list of int; list of target index
                target_position_recall-N:
                    es_target_index_rank_list: list of list of int; list of list of ranked target index, from closest to least closest
        """
        semantic_tag_unique_list = list(set(self.node_semantic_tag_list))
        semantic_tag_total_count = np.array([self.node_semantic_tag_list.count(semantic_tag) for semantic_tag in semantic_tag_unique_list])

        es_semantic_tag_list = []
        es_target_index_list = []
        es_target_index_rank_list = []
        query_feature_dist = []
        node_index_list = []
        for query_image_path in query_image_path_list:
            query_image = cv2.imread(query_image_path)
            query_image_batch = self.process_batch([query_image])
            query_rolling_features_list = self.sess.run(self.rolling_features_list,
                    feed_dict = {self.images_placeholder: query_image_batch})

            metric_distance_list = []
            for node_index in range(self.node_number):
                node_metric_feature = self.node_metric_feature_list[node_index]
                rolling_distance_list = []
                for query_rolling_features in query_rolling_features_list:
                    rolling_distance = np.sqrt(np.sum(np.square(node_metric_feature - query_rolling_features)))
                    rolling_distance_list.append(rolling_distance)
                metric_distance = min(rolling_distance_list)
                metric_distance_list.append(metric_distance)
            metric_distance_list = np.array(metric_distance_list)

            query_feature_dist.append(metric_distance_list)


            node_index_rank = np.argsort(metric_distance_list)
            node_index_list.append(node_index_rank[0])
            es_target_index_rank = [self.node_id_list[node_index] for node_index in node_index_rank]
            es_target_index_rank_list.append(es_target_index_rank)
            es_target_index_list.append(es_target_index_rank[0])
            es_semantic_tag_list.append(set([self.node_semantic_tag_list[node_index_rank[0]]]))

        if evaluation_method == 'find_target_position':
            print("TARGET INDEX: "+str(es_target_index))
            print("semantic tag: "+str(self.node_semantic_tag_list[es_target_index]))
            print("\n")
            return es_semantic_tag_list, es_target_index_list

        elif evaluation_method == 'target_position_precision-error':
            return es_target_index_list
        elif evaluation_method == 'target_position_distance-accuracy':
            return es_target_index_list
        elif evaluation_method == 'target_position_recall-N':
            return es_target_index_rank_list
        elif evaluation_method == 'target_position_precision-recall':
            return [node_index_list, query_feature_dist]
        else:
            raise ValueError("unknown evaluation method")


if __name__ == '__main__':
    # the json path to run localization
    train_path = '/home/joehuang/Desktop/Final_Test/Scene_0-dem1-omni.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_node_distance', type=float,
        default=0.7,
        help='when forming node in demonstration file, minimum distance to set another node')

    parser.add_argument('--feature_size', type=int,
        default=512,
        help='feature size')
    parser.add_argument('--conv_initialization', type=str,
        #default = 'random',
        #default = 'pretrain',
        default = 'trained',
        help='model directory')
    parser.add_argument('--pretrain_path', type=str,
        default='/home/joehuang/Desktop/Node_ML/evaluation/node_ver1/googlenet_places365.npy',
        help='if random initialized, then load this pretrain model')
    parser.add_argument('--model_path', type=str,
        default='/home/joehuang/Desktop/Node_ML/evaluation/node_ver4/model_V1',
        help='model directory')
    parser.add_argument('--model_checkpoint', type=str,
        default='latest', # latest checkpoint
        #default ='0.0001-4000',
        help='model directory')
    parser.add_argument('--random_initialize_fc_layer', type=bool,
        default=False,
        help='if you load model checkpoint, do you want to random initialize fc layers?')
    FLAGS, _ = parser.parse_known_args()
    localization = Localization_Evaluation(FLAGS)

    localization.demonstrate(train_path)

    query_image_path_list = []
    position_list = []
    angle_list = []
    with open('./mytest.txt', 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            else:
                info = line[:-1].split(' ')
                query_image_path_list.append(str(info[-1]))
                position_list.append([float(info[1]), float(info[2]), float(info[3])])
    demonstration_subdir = train_path[:-5].split('/')[-1]
    result = localization.prediction(query_image_path_list, 'find_target_position', demonstration_subdir)
    print("prediction complete! To test algorithm correctness, please use metric")
    pdb.set_trace()
