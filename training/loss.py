import tensorflow as tf
import numpy as np
import pdb
import math
import random
import itertools

def create_position_np(position_np):
    # create position numpy array (np.2darray), (B,2)
    return position_np

def create_pospair_index_np(batch_size, positive_size, position_np):
    """
    random pick positive pair index, remember that first half of batch are in the same room, pick from them
    Input:
        batch_size: int; the batch size
        positive_size: int; the positve size
    Return:
        pospair_index_np: np.2darray (positive_size, 2); the random picked positive pair indexes
    """
    #pospair_index = np.array(random.sample(list(itertools.combinations(range(batch_size / 2), 2)), positive_size))
    bin_num = [4,4,4,4,4,4,4,4]
    pospair_index_list = []
    control_distances = [(0,0.5), (0.5, 1.0), (1.0, 1.5), (1.5,2.0), (2.0,2.5), (2.5,3.0),(3.0,4.0),(4.0,float('inf'))]
    P, D = position_np.shape
    half_position_np = position_np[:P/2,:]
    col_expand = np.tile(np.reshape(half_position_np, (P/2,1,D)), (1,P/2,1))
    row_expand = np.tile(np.reshape(half_position_np, (1,P/2,D)), (P/2,1,1))
    pos_dist = np.sqrt(np.sum(np.square(col_expand - row_expand), 2))
    all_pairs = np.array(list(itertools.combinations(range(P/2),2)))
    np.random.shuffle(all_pairs)

    leftpair_index_list = []
    for pair in all_pairs:
        pair_distance = pos_dist[pair[0],pair[1]]
        for index in range(len(bin_num)):
            min_distance = control_distances[index][0]
            max_distance = control_distances[index][1]
            if pair_distance >= min_distance and pair_distance <= max_distance:
                if bin_num[index] != 0:
                    bin_num[index] -= 1
                    pospair_index_list.append(pair)
                else:
                    leftpair_index_list.append(pair)
                break
        if len(pospair_index_list) == positive_size:
            break
    if len(pospair_index_list) != positive_size:
        leftpair_number = positive_size - len(pospair_index_list)
        pospair_index_list += random.sample(leftpair_index_list, leftpair_number)
    pospair_index = np.array(pospair_index_list)

    distribution = np.zeros(len(control_distances))
    for pospair in pospair_index:
        distance = np.sqrt(np.sum(np.square(position_np[pospair[0]] - position_np[pospair[1]])))
        for idx in range(len(control_distances)):
            control_distance = control_distances[idx]
            min_dis = control_distance[0]
            max_dis = control_distance[1]
            if distance >= min_dis and distance <= max_dis:
                distribution[idx] += 1
                break
    return pospair_index, np.array(distribution)


def create_rot_gauss_np(rotation_labels_np, B, L, sigma_R):
    """
    generate rotation gaussian numpy array
    Input:
        rotation_labels_np: [B] np.1darray; the rotation labels of images
        B: int; batch size
        L: int; numbers of rotation
        sigma_R: float; the rotation sigma
    Return:
        rot_gauss: np.3darray [B,B,L] float; the rotation gaussian np array
    """
    rotation_row_expand = np.tile(np.reshape(rotation_labels_np, [1,B]), [B,1])
    rotation_col_expand = np.tile(np.reshape(rotation_labels_np, [B,1]), [1,B])
    rotation_mat = rotation_col_expand - rotation_row_expand #[r,c]=a[r]-a[c]
    rot = np.tile(np.reshape(rotation_mat, [B,B,1]), [1,1,L])

    rot_array = np.tile(np.reshape(np.arange(L) * 360. / L, [1,1,L]), [B,B,1])
    rot_array = np.absolute(rot_array - rot) % 360.
    rot_array = np.minimum(rot_array, 360. - rot_array)
    rot_gauss = np.exp(-0.5 * np.square(rot_array / sigma_R))

    rot_sum = np.tile(np.reshape(np.sum(rot_gauss, axis=2), [B,B,1]), [1,1,L])
    rot_gauss = rot_gauss / rot_sum
    return rot_gauss


def loss(predict_branches, positions, rot_gauss, pospair_index, margin, sigma_R, negative_tolerance):
    """
    calculate loss from prediction matrix, margin, label matrix

    Input:
        prdict_branches: list of tensor float[batch_num, width, feature_size]; the prediction branches coming out from network, ex: list of 20 [64,20,512] tensor
        positions: tensor [B,2] float; the (x,z) position for each query
        rot_gauss: tensor [B,B,L] float; the gaussian weight for rotating different angles, ex:[128,128,20]
        pospair_index: tensor [positive_pairs, 2] int; the positive pair index, ex: [[1,2],[3,2]]
        margin: float; the margin defined in loss function
        sigma_R: float; the sigma of rotation for positive pair, in degrees
        negative_tolerance: float; the negative distance have to be greater than this + positive distance to be count as negative
    Return:
        loss: tensor[float]; the calculated loss, is used in future training process
    """
    # calculate label matrix
    predict_branches = [predict_branches[0]] #NOTE: for camera ready no R
    L = len(predict_branches)
    B, W, C = predict_branches[0].get_shape().as_list()

    # calculate feature distance matrix [B,B,L], L is length of predict_branches
    # b1, b2, l1 is the distance of feature[b2] rotate left l1 nd feature[b1]
    root_branch = predict_branches[0]
    col_expand = tf.tile(tf.reshape(root_branch, [B,1,W*C]), [1,B,1])
    dist_mat_list = []
    for predict_branch in predict_branches:
        row_expand = tf.tile(tf.reshape(predict_branch, [1,B,W*C]), [B,1,1])
        dist_mat_branch = col_expand - row_expand
        dist_mat_branch = tf.sqrt(tf.reduce_sum(tf.square(dist_mat_branch), 2) + tf.constant(1e-14))
        dist_mat_list.append(tf.reshape(dist_mat_branch, [B,B,1]))
    dist_mat = tf.concat(dist_mat_list, axis=2) / L #NOTE: check loss multiplier (/L)

    # calculate position distance matrix [B,B]
    pos_col_expand = tf.tile(tf.reshape(positions, [B,1,2]), [1,B,1])
    pos_row_expand = tf.tile(tf.reshape(positions, [1,B,2]), [B,1,1])
    pos_dist_mat = tf.sqrt(tf.reduce_sum(tf.square(pos_col_expand - pos_row_expand), 2) + tf.constant(1e-14))

    # positive index
    pos_pairs = pospair_index
    pos_pairs_num = tf.shape(pos_pairs)[0]

    # initialized all for loop variables
    total_loss = tf.constant(0.)
    # start going through all positive pairs
    for index in range(pos_pairs.get_shape().as_list()[0]):
        # calculate loss when it is positive pair
        row = tf.cast(pos_pairs[index][0], tf.int32)
        col = tf.cast(pos_pairs[index][1], tf.int32)
        pos_dist_thresh = pos_dist_mat[row][col] + tf.constant(negative_tolerance)

        mask_row = tf.cast(tf.greater_equal(pos_dist_mat[row], pos_dist_thresh), tf.float32)
        mask_row = tf.tile(tf.reshape(mask_row, [B,1]), [1,L])
        mask_col = tf.cast(tf.greater_equal(pos_dist_mat[col], pos_dist_thresh), tf.float32)
        mask_col = tf.tile(tf.reshape(mask_col, [B,1]), [1,L])

        row_neg = dist_mat[row]
        col_neg = dist_mat[col]

        D_neg = tf.reduce_sum(tf.multiply(tf.exp(margin - row_neg), mask_row)) + tf.reduce_sum(tf.multiply(tf.exp(margin - col_neg), mask_col))

        D_pos_branch = dist_mat[row][col]
        D_pos = tf.reduce_sum(tf.multiply(D_pos_branch, rot_gauss[row][col]))

        J = tf.square(tf.maximum(0.0, (tf.log(D_neg) + D_pos)))
        total_loss = total_loss + J

    total_loss = total_loss / (2. * tf.cast(pos_pairs_num, tf.float32))
    return total_loss
