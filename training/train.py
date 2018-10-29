import numpy as np
import cv2
import os
import pdb
import loss
from model import CaffeNet
from datetime import datetime
import argparse
from reader import Reader
from triplet_reader import Triplet_Reader
import time
import math
import sys
sys.path.append("../../")
from interface import Localization_Training_Interface
import tensorflow as tf

"""
class for only training
including create reader
no queue, both training and validatoin are placeholder
"""

class Localization_Training(Localization_Training_Interface):
    """
    class for training localization network
    """
    def __init__(self, FLAGS):
        """
        create the network and training, validation reader functions
        Input:
            FLAGS:
                gpu_number: int; the gpu number
                positive_size: int; the positive pair size to be random chosen in each batch
                total_batch_number: int; batches to create in reader file
                tolerance_margin: float; how much negative distance should be greater than positive distance
                batch_size: int; training and validation batch size
                feature_size: int; size of feature output from the network
                data_dir: string; data set directory
                training_data_filename: string; training data filename, e.g. CG_train.txt
                validation_data_filename: string; validation data filename, e.g. CG_test.txt
        """
        gpu_number = FLAGS.gpu_number
        self.positive_size = FLAGS.positive_size
        total_batch_number = FLAGS.total_batch_number
        tolerance_margin = FLAGS.tolerance_margin
        pick_same_room = True # when validating, should we pick same room to generate negative pair?
        batch_size = FLAGS.batch_size
        feature_size = FLAGS.feature_size
        data_dir = FLAGS.data_dir
        training_data_filename = FLAGS.training_data_filename
        validation_data_filename = FLAGS.validation_data_filename
        self.control_distances = [(0,0.5), (0.5, 1.0), (1.0, 1.5), (1.5,2.0), (2.0,2.5), (2.5,3.0),(3.0,4.0),(4.0,float('inf'))]

        self.sigma_R = 30
        self.margin = 1.0

        # imagenet mean
        #images_mean = tf.constant([104./127.5 - 1.0, 117./127.5 -1.0, 123./127.5 - 1.0])
        # place365 mean
        images_mean = tf.constant([104./127.5 - 1.0, 113./127.5 -1.0, 117./127.5 - 1.0])
        self.images_placeholder = tf.placeholder(tf.float32,
                                         shape=(batch_size,384,640,3))
        images = self.images_placeholder/127.5 - 1.0 - images_mean
        # the positions
        self.position_placeholder = tf.placeholder(tf.float32,
                                         shape=(batch_size, 2))
        # rotational gauss placeholder
        self.rot_gauss_placeholder = tf.placeholder(tf.float32,
                                         shape=(batch_size, batch_size, 20))
        # positive pair index placeholder (chosen randomly using numpy)
        self.pospair_index_placeholder = tf.placeholder(tf.int32,
                                         shape=(self.positive_size, 2))

        print('Building graph...')
        with tf.device('/gpu:'+str(gpu_number)):
            self.net = CaffeNet({'data': images})
            # before dropout
            conv_features = self.net.get_output()
            B, H, W, C = conv_features.get_shape().as_list()
            F = feature_size
            self.B = B
            self.W = W
            self.C = C
            conv_features = tf.reshape(conv_features, [B, H*W, C])
            t_list = tf.split(conv_features, num_or_size_splits=W, axis=1)
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
            self.lifted_loss = loss.loss(self.rolling_features_list, self.position_placeholder,
                    self.rot_gauss_placeholder, self.pospair_index_placeholder,
                    self.margin, self.sigma_R, tolerance_margin)

            reg_loss = 0
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                #print var.name
                reg_loss += tf.nn.l2_loss(var)

            self.total_loss = 0.0008 * reg_loss + self.lifted_loss
            self.training_summary = tf.summary.scalar('training_loss', self.total_loss)
            self.validation_summary = tf.summary.scalar('validation_loss', self.total_loss)

            self.validation_accuracy_placeholder = tf.placeholder(tf.float32, shape=())
            self.training_accuracy_placeholder = tf.placeholder(tf.float32, shape=())
            validation_accuracy = tf.get_variable('validation_accuracy', [], trainable=False)
            training_accuracy = tf.get_variable('training_accuracy', [], trainable=False)
            self.validation_assign_op = tf.assign(validation_accuracy, self.validation_accuracy_placeholder)
            self.training_assign_op = tf.assign(training_accuracy, self.training_accuracy_placeholder)
            self.validation_accuracy_summary = tf.summary.scalar('validation_accuracy', validation_accuracy)
            self.training_accuracy_summary = tf.summary.scalar('training_accuracy', training_accuracy)

            # Adam optimizer
            global_step = tf.get_variable('global_step', [],
                                             initializer = tf.constant_initializer(0), trainable = False)
            opt = tf.train.MomentumOptimizer(0.00001, 0.9)
            grads_and_vars = opt.compute_gradients(self.total_loss)
            self.train_op = opt.apply_gradients(grads_and_vars, global_step = global_step)
            print('Finish building graph')

        print("start creating reader objects")
        #create training batch reader object
        parser = argparse.ArgumentParser()
        train_FLAGS, _ = parser.parse_known_args()
        train_FLAGS.data_dir = data_dir
        train_FLAGS.batch_size = batch_size
        train_FLAGS.total_batch_number = total_batch_number
        train_FLAGS.tolerance_margin = tolerance_margin
        train_FLAGS.pick_same_room = pick_same_room
        train_FLAGS.data_filename = training_data_filename
        self.training_reader = Reader(train_FLAGS)
        self.training_triplet_reader = Triplet_Reader(train_FLAGS, self.control_distances)

        # create validation batch reader object
        parser = argparse.ArgumentParser()
        validation_FLAGS, _ = parser.parse_known_args()
        validation_FLAGS.data_dir = data_dir
        validation_FLAGS.batch_size = batch_size
        validation_FLAGS.total_batch_number = total_batch_number
        validation_FLAGS.tolerance_margin = tolerance_margin
        validation_FLAGS.pick_same_room = pick_same_room
        validation_FLAGS.data_filename = validation_data_filename
        self.validation_reader = Reader(validation_FLAGS)
        self.validation_triplet_reader = Triplet_Reader(validation_FLAGS, self.control_distances)
        print("Finish creating reader objects")

    def train(self, model_path, save_dir='./model', summary_dir='/home/joehuang/Desktop/Node_ML/log', iters=20000, validation_step=200,
                evaluate_training_accuracy = True, save_step=1000):
        """
        training the model, calculate validation accuracy using triplet
        Input:
            model_path: string; the absolute path of the pre-trained model
            save_dir: string; the directory to save the checkpoints
            summary_dir: string; the directory to save tensorflow summary
            iters: int; the total training iterations
            validation_step: int; numbers of iterations per validation
            evaluate_training_accuracy: bool; evaluate training accuracy or not
            save_step: int; numbers of iterations per saving
        """
        # model_path, iters, validation_step, save_step, save_dir
        saver = tf.train.Saver(max_to_keep = 20)
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(summary_dir,graph_def=sess.graph_def)
            summary_writer_list = [summary_writer]
            for sw_i in range(1,len(self.control_distances)):
                now_dir = os.path.join(summary_dir, 'plot_'+str(sw_i))
                if os.path.isdir(now_dir):
                    os.mkdir(now_dir)
                summary_writer_list.append(tf.summary.FileWriter(now_dir,graph_def=sess.graph_def))

            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print('Loading pre-trained model...')
            self.net.load(model_path, sess, True)
            print('pre-trained model done loading')
            total_distbin = np.zeros(len(self.control_distances))

            for step in range(iters):
                training_image_batch, training_class_label_batch, training_rotation_batch, training_position_batch = self.training_reader.next_batch()
                training_rot_gauss_np = loss.create_rot_gauss_np(training_rotation_batch, self.B, self.W, self.sigma_R)
                training_position_np = loss.create_position_np(training_position_batch)
                training_pospair_index_np, dist_bin = loss.create_pospair_index_np(self.B, self.positive_size, training_position_np)
                total_distbin += dist_bin
                _, losses, summary = sess.run(
                        [self.train_op, self.total_loss, self.training_summary],
                        feed_dict={self.images_placeholder:training_image_batch,
                                self.position_placeholder: training_position_np,
                                self.rot_gauss_placeholder: training_rot_gauss_np,
                                self.pospair_index_placeholder: training_pospair_index_np})
                summary_writer.add_summary(summary, step)

                # show losses
                if step != 0 and step % 1 == 0:
                    format_str = ('%s: step %d, loss = %.6f')
                    print(format_str % (datetime.now(), step, losses))

                if (step % validation_step == 0):
                    print total_distbin
                    validation_image_batch, validation_class_label_batch, validation_rotation_batch, validation_position_batch = self.validation_reader.next_batch()
                    validation_rot_gauss_np = loss.create_rot_gauss_np(validation_rotation_batch, self.B, self.W, self.sigma_R)
                    validation_position_np = loss.create_position_np(validation_position_batch)
                    validation_pospair_index_np, dist_bin = loss.create_pospair_index_np(self.B, self.positive_size, validation_position_np)
                    losses, summary = sess.run(
                            [self.total_loss, self.validation_summary],
                            feed_dict={self.images_placeholder:validation_image_batch,
                                    self.position_placeholder: validation_position_np,
                                    self.rot_gauss_placeholder: validation_rot_gauss_np,
                                    self.pospair_index_placeholder: validation_pospair_index_np})
                    summary_writer.add_summary(summary, step)
                    print('validation: step {}, validation loss = {:.6}'.format(step, losses))

                    # triplet reader only output image, batch [0]: anchor, [1]: positive, [2]: negative
                    accuracy_iteration = 2
                    # list of list of float, outer list is control distance index,inner list is accuracy iteration
                    validation_control_distance_accuracy_list = []
                    training_control_distance_accuracy_list = []
                    for control_distance_index in range(len(self.control_distances)):
                        validation_accuracy_list = []
                        training_accuracy_list = []
                        for index in range(accuracy_iteration):
                            anchor_image_batch, positive_image_batch, negative_image_batch, rotation_batch = self.validation_triplet_reader.next_batch(control_distance_index)
                            anchor_feature = sess.run(self.rolling_features_list, feed_dict={self.images_placeholder: anchor_image_batch})
                            positive_feature = sess.run(self.rolling_features_list, feed_dict={self.images_placeholder: positive_image_batch})
                            negative_feature = sess.run(self.rolling_features_list, feed_dict={self.images_placeholder: negative_image_batch})
                            correct_number = self.validation_triplet_reader.evaluate_correct_number(anchor_feature, positive_feature, negative_feature, rotation_batch)
                            if index == 0 and control_distance_index == 0:
                                print("validation")
                                rotation_error = self.validation_triplet_reader.evaluate_rotation_error(anchor_feature, positive_feature, negative_feature, rotation_batch)

                            all_number = self.B
                            accuracy = float(correct_number) / float(all_number)
                            validation_accuracy_list.append(accuracy)

                            # triplet reader only output image, batch [0]: anchor, [1]: positive, [2]: negative
                            anchor_image_batch, positive_image_batch, negative_image_batch, rotation_batch = self.training_triplet_reader.next_batch(control_distance_index)
                            anchor_feature = sess.run(self.rolling_features_list, feed_dict={self.images_placeholder: anchor_image_batch})
                            positive_feature = sess.run(self.rolling_features_list, feed_dict={self.images_placeholder: positive_image_batch})
                            negative_feature = sess.run(self.rolling_features_list, feed_dict={self.images_placeholder: negative_image_batch})
                            correct_number = self.training_triplet_reader.evaluate_correct_number(anchor_feature, positive_feature, negative_feature, rotation_batch)
                            if index == 0 and control_distance_index == 0:
                                print("training")
                                rotation_error = self.training_triplet_reader.evaluate_rotation_error(anchor_feature, positive_feature, negative_feature, rotation_batch)

                            all_number = self.B
                            accuracy = float(correct_number) / float(all_number)
                            training_accuracy_list.append(accuracy)
                        validation_control_distance_accuracy_list.append(validation_accuracy_list)
                        training_control_distance_accuracy_list.append(training_accuracy_list)

                    validation_control_distance_accuracy = [sum(validation_accuracy_list) / accuracy_iteration for validation_accuracy_list in validation_control_distance_accuracy_list]
                    training_control_distance_accuracy = [sum(training_accuracy_list) / accuracy_iteration for training_accuracy_list in training_control_distance_accuracy_list]
                    for record_index in range(len(self.control_distances)):
                        print("summary only record accuracy of target distance from {} to {}".format(self.control_distances[record_index][0], self.control_distances[record_index][1]))
                        sess.run(self.validation_assign_op, feed_dict={self.validation_accuracy_placeholder: validation_control_distance_accuracy[record_index]})
                        summary = sess.run(self.validation_accuracy_summary)
                        summary_writer_list[record_index].add_summary(summary, step)

                        sess.run(self.training_assign_op, feed_dict={self.training_accuracy_placeholder: training_control_distance_accuracy[record_index]})
                        summary = sess.run(self.training_accuracy_summary)
                        summary_writer_list[record_index].add_summary(summary, step)

                    print('accuracy in range' + str(self.control_distances))
                    print('training accuracy: ' + str(training_control_distance_accuracy))
                    print('validation accuracy: ' + str(validation_control_distance_accuracy))
                # Save the model checkpoint periodically
                if (step != 0) and (step % save_step == 0):
                    path = os.path.join(save_dir, str(0.0001))
                    saver.save(sess, path, global_step = step)

            print('Done training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_number', type=int, default=0, help='gpu number')
    parser.add_argument('--positive_size', type=int, default=32, help='the positive pair size to be random chosen in each batch')
    parser.add_argument('--total_batch_number', type=int, default=1500, help='number of batches to be created every time in reader file')
    parser.add_argument('--tolerance_margin', type=float, default=0.5, help='how much negative distance should be greater than positive distance to be count as positive')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--feature_size', type=int, default=512, help='feature size')
    parser.add_argument('--data_dir', type=str, default='/home/joehuang/Desktop/room_data_continuous_omni', help='data directory')
    parser.add_argument('--training_data_filename', type=str, default='CG_train2.txt', help='filename containing training data list')
    parser.add_argument('--validation_data_filename', type=str, default='CG_test2.txt', help='filename containing validation data list')
    FLAGS, unparsed = parser.parse_known_args()
    localization = Localization_Training(FLAGS)
    localization.train('./googlenet_places365.npy', './camera_ready_no_R', '/home/joehuang/Desktop/Node_ML/log/camera_ready_no_R')

