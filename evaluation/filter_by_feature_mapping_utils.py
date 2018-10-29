from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import cv2
import random
import pdb

import warnings
warnings.filterwarnings("ignore")

def feature_matching(tensor_in, top_k=10):
    '''
    FUNC: Given 2 feature map, find correspdent points.
    Each feature vector is along C axis, and we compute L1
    distance between 2 feature vectors for comparison.
    Arguments:
        - tensor_in: a numpy array of shape (B,H,W,C), we
          compare 1st feature map tensor_in[0] (query) to 
          rest of the feature map tensor_in[1:] (others)
        - top_k: during each comparison, find top_k'th minimum
          distance between each pairs of points in feature map
          as correspondent points.
    Returns:
        - dist_mat_list: a list of length B-1 (there are B-1 
          comparisons), with each element as a (H*W,H*W) nparray,
          also called distance matrix. The (i,j)'th entry of a
          distance matrix represents distance between (i/W,i%W)'th
          point in "others" and (j/W,j%W)'th point in "query".
        - pair_list: a list of length B-1, with each element as
          a (top_k,5) nparray (there are top_k matched points), 
          representing (q_i,q_j,o_i,o_j,dist). q_i and q_j refer to 
          correspondent point is at (q_i,q_j) in "query" and so so 
          o_i, o_j in "others". dist means the distance between the 
          the particular pair.
        - top_k_dist_list: a list of length B-1, with each element
          as the top k'th minimum distance among comparisons between
          the "query" and a "others"
    '''
    B, H, W, C = tensor_in.shape
    new_dim = H*W
    tensor_in = np.reshape(tensor_in, (B,new_dim,C))
    # seperate query image and others
    t_list = np.split(tensor_in, B)
    query = t_list[0] # ((H*W),C)
    others = t_list[1:] # (B-1) list with each shape as ((H*W),C)
   
    # compute distance matrix
    dist_mat_list = [] # (B-1) list with shape ((H*W),(H*W))
    pair_list = []
    top_k_dist_list = []
    query = np.tile(query, (new_dim,1,1)) # of shape (H*W,H*W,C)
    for o in others:
        # make mesh grid between query and o
        o = np.tile(o, (new_dim,1,1)) # of shape (H*W,H*W,C)
        o = o.transpose((1,0,2)) # transpose for forming mesh grid
        
        # compute distance matrix
        dist_mat = np.sum(np.absolute(query-o), axis=2) # absolute difference
        dist_mat_list.append(dist_mat)

        # find top_k in distance matrix
        top_k_min_idx = np.argpartition(dist_mat, top_k, axis=None)[:top_k] # top_k but unsorted
        top_k_min_i = (top_k_min_idx/new_dim).astype(np.int32)
        top_k_min_j = (top_k_min_idx%new_dim).astype(np.int32)
        top_k_dist = dist_mat[top_k_min_i,top_k_min_j]
        # top_k found above are actually unsorted, we need to further sort
        # the top_k length vector
        top_k_sort_idx = np.argsort(top_k_dist)
        top_k_min_i = top_k_min_i[top_k_sort_idx]
        top_k_min_j = top_k_min_j[top_k_sort_idx]
        top_k_dist = top_k_dist[top_k_sort_idx]

        # use top_k minimum idices to find pairs between 2 images
        q_i = (top_k_min_j/W).astype(np.int32)
        q_j = (top_k_min_j%W).astype(np.int32)
        o_i = (top_k_min_i/W).astype(np.int32)
        o_j = (top_k_min_i%W).astype(np.int32)
        pair = np.array([q_i,q_j,o_i,o_j]).transpose()

        # append to lists
        dist_mat_list.append(dist_mat)
        pair_list.append(pair)
        top_k_dist_list.append(top_k_dist)

    return dist_mat_list, pair_list, top_k_dist_list 

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min()-0.1:x.max()+0.1:xbins, 
                      y.min()-0.1:y.max()+0.1:ybins]

    xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T
    xy_train  = np.vstack([x, y]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

def analyze_pairs(pair_list, top_k_dist_list, dist_thresh, min_pts, cons_thresh, bandwidth=0.5):
    def trans_func(p, mode='Cartesian'):
        qi, qj, oi, oj = p
        dist_i = oi - qi
        dist_j = oj - qj
        if mode=='Cartesian':
            return np.array([dist_i, dist_j])
        elif mode=='Polar':
            return np.array([math.sqrt(dist_i**2+dist_j**2), atan2(dist_i,dist_j)])
        else:
            raise ValueError('mode must be Cartesian or Polar in trans_func()')

    top_k = top_k_dist_list[0].shape[0]
    filtered_pair_list = []
    fbool_list = []
    trans_list = []
    density_list = []
    cons_list = []
    for pairs, top_k_dist in zip(pair_list,top_k_dist_list):
        # filter out pairs with distance larger than a threshold
        # filtered pairs will have value of -1
        filtered_pairs = -1 * np.ones_like(pairs, dtype=np.int16)
        trans = np.zeros((top_k,2))
        fbool = np.zeros((top_k,), dtype=np.bool)
        fbool_count = 0
        for idx, dist in enumerate(top_k_dist):
            if dist<=dist_thresh:
                filtered_pairs[idx] = pairs[idx]
                fbool[idx] = True
                fbool_count += 1
                
            # compute transition of the 2 points for each pairs.
            # there are 2 representation, 1.Cartesian 2.polar coordinate
            trans[idx] = trans_func(pairs[idx])
    
        # regress a linear function based on transition of correspondent points
        if fbool_count>=min_pts:
            data_in = trans[fbool]
            i = data_in[:,0]
            j = data_in[:,1]
            ii, jj, zz = kde2D(i, j, bandwidth)
            density = [ii,jj,zz]
            #TODO: err undefined
            err = 9999
        else:
            density = None
            err = 9999

        # if linear regression error lower than a threshold, view
        # this set of correspondent points as "consensus pairs"
        if err<cons_thresh:
            cons_list.append(True)
        else:
            cons_list.append(False)

        # append to lists
        filtered_pair_list.append(filtered_pairs)
        trans_list.append(trans)
        density_list.append(density)
        fbool_list.append(fbool)
    
    return filtered_pair_list, trans_list, density_list, fbool_list, cons_list

def visualize_pairs(pair_list, H, W, colors, mul=50):
    '''
    FUNC: visualize pairs, this function works with feature_matching()
    Arguments:
        - pair_list: return data pair_list from feature_matching()
        - H: height of feature map
        - W: width of feature map
        - colors: a list of tuple, each tuple is a color
        - mul: multiplier of creating images for visualization
    Returns:
        - show_q_list: a list of images of shape (H*mul,W*mul,3) showing 
          correspondent point in "query"
        - show_o_list: a list of images of shape (H*mul,W*mul,3) showing 
          correspondent point in "others"
    '''
    n_colors = len(colors)
    assert pair_list[0].shape[0]==n_colors

    show_q_list = []
    show_o_list = []
    for i, pairs in enumerate(pair_list):
        if not (-1 in pairs):
            vis_q = np.zeros((H,W,3), dtype=np.uint8)
            vis_o = np.zeros((H,W,3), dtype=np.uint8)
            show_q = np.zeros((mul*H,mul*W,3), dtype=np.uint8)
            show_o = np.zeros((mul*H,mul*W,3), dtype=np.uint8)
            for j in range(n_colors):
                qi, qj, oi, oj = pairs[j]
                frac = int(j/n_colors*mul)
                if (vis_q[qi,qj]==(0,0,0)).all():
                    vis_q[qi,qj] = colors[j]
                    show_q[qi*mul:(qi+1)*mul, qj*mul:(qj+1)*mul] = colors[j]
                else:
                    show_q[qi*mul:(qi+1)*mul, qj*mul+frac:(qj+1)*mul] = colors[j]
                if (vis_o[oi,oj]==(0,0,0)).all():
                    vis_o[oi,oj] = colors[j]
                    show_o[oi*mul:(oi+1)*mul, oj*mul:(oj+1)*mul] = colors[j]
                else:
                    show_o[oi*mul:(oi+1)*mul, oj*mul+frac:(oj+1)*mul] = colors[j]

            show_q_list.append(show_q.copy())
            show_o_list.append(show_o.copy())
        else:
            show_q_list.append(None)
            show_o_list.append(None)

    return show_q_list, show_o_list

def visualize_pairs_arrowed_line(pair_list, H, W, colors, mul=50):
    '''
    FUNC: visualize pairs, using arrowed line to represent transition
    between correspondent points in a pair
    Arguments:
        - pair_list: return data pair_list from feature_matching()
        - H: height of feature map
        - W: width of feature map
        - colors: a list of tuple, each tuple is a color
        - mul: multiplier of creating images for visualization
    Returns:
        - show_t_list: a list of images of shape (H*mul,W*mul,3) showing 
          transition of correspondent point between "query" and "others"
    '''
    n_colors = len(colors)
    assert pair_list[0].shape[0]==n_colors

    def sample_ij(range_L, range_H):
        return random.randint(int(range_L), int(range_H))

    show_t_list = []
    line_half_w = 2
    for pairs in pair_list:
        if not (-1 in pairs): # exclude omitted elements
            # create grid
            show_t = np.zeros((H*mul,W*mul,3), dtype=np.uint8)
            for k in range(mul,H*mul,mul):
                show_t[k-line_half_w:k+line_half_w,:] = (255,255,255)
            for k in range(mul,W*mul,mul):
                show_t[:,k-line_half_w:k+line_half_w] = (255,255,255)
            # draw arrowed line
            for j in range(n_colors):
                qi, qj, oi, oj = pairs[j]
                show_qi = sample_ij((qi+0.3)*mul, (qi+0.7)*mul)
                show_qj = sample_ij((qj+0.3)*mul, (qj+0.7)*mul)
                show_oi = sample_ij((oi+0.3)*mul, (oi+0.7)*mul)
                show_oj = sample_ij((oj+0.3)*mul, (oj+0.7)*mul)
                cv2.line(show_t, (show_qj,show_qi), (show_oj,show_oi), colors[j], 2)
                cv2.circle(show_t, (show_qj,show_qi), 4, colors[j], 3)
            # append to list
            show_t_list.append(show_t)
        else:
            show_t_list.append(None)

    return show_t_list

def create_random_colors(n):
    '''
    FUNC: create n random colors
    Arguments:
        - n: a scalar
    Returns:
        - colors: a list of tuple, with each tuple as one color
    '''
    colors = []
    for _ in range(n):
        colors.append(tuple(np.random.randint(0, 255, (3))))

    return colors

def create_color_bar(colors, sz=100):
    '''
    FUNC: create image of color bar
    Arguments:
        - colors: a list of tuple with each element as a color
        - sz: size of a block representing one color in the color bar
    Returns:
        - color_bar: an image of color bar, shape=(sz*n_colors,sz,3)
    '''
    n_colors = len(colors)
    color_bar = np.zeros((sz*n_colors,sz,3),dtype=np.uint8)
    for i in range(n_colors):
        color_bar[i*sz:(i+1)*sz] = colors[i]

    return color_bar

def test1():
    print('This is test 1')
    n_others = 5
    H, W, C = (7,5,3)
    top_k = 10
    n_similar = 2
    sim_mu = 0
    sim_sig = 0.5
    dis_mu = 10
    dis_sig = 1
    colors = create_random_colors(top_k)

    # create test matrix with shape (B,H,W,C)
    # with the first member in batch is the query image and 
    # we gonna compare query image to the rest of the images
    query = np.random.normal(sim_mu, sim_sig, (H,W,C))
    others = np.random.normal(dis_mu, dis_sig, (n_others,H,W,C))
    query_like = np.random.normal(sim_mu, sim_sig, (n_others,n_similar,n_similar,C))

    test_mat = np.zeros((n_others+1,H,W,C))
    test_mat[0] = query
    test_mat[1:] = others
    test_mat[1:,:n_similar,:n_similar,:] = query_like

    # perform visual matching, output may be a list of
    # distance matrix with length B-1, with each matrix
    # of shape (H*W,H*W)
    dist_mat_list, pair_list, top_k_dist_list = feature_matching(test_mat, top_k)
    fpair_list, trans_list, density_list, fbool_list, cons_list = analyze_pairs(pair_list, 
                                                                             top_k_dist_list, 
                                                                             999,
                                                                             4,
                                                                             5)

    # visualization
    show_q_list, show_o_list = visualize_pairs(pair_list, H, W, colors)
    show_t_list = visualize_pairs_arrowed_line(pair_list, H, W, colors)
    color_bar = create_color_bar(colors)
    cv2.imshow('colorbar', color_bar)
    for i in range(len(show_q_list)):
        show_q = show_q_list[i]
        show_o = show_o_list[i]
        show_t = show_t_list[i]
        filtered_pairs = fpair_list[i]
        trans = trans_list[i]
        density = density_list[i]
        fbool = fbool_list[i]

        inliers_i = trans[:,0][fbool]
        inliers_j = trans[:,1][fbool]
        outliers_i = trans[:,0][np.invert(fbool)]
        outliers_j = trans[:,1][np.invert(fbool)]

        fig = plt.figure(1)
        ax = fig.add_subplot(211)
        if density is not None:
            ax.pcolormesh(density[0],density[1],density[2])
        if inliers_i.shape[0]>0:
            ax.scatter(inliers_i, inliers_j, color='white')
        if outliers_i.shape[0]>0:
            ax.scatter(outliers_i, outliers_j, color='red')
        plt.show()

        cv2.imshow('query',show_q)
        cv2.imshow('others',show_o)
        cv2.imshow('transition',show_t)
        key = cv2.waitKey(0)
        if key&0xFF==ord('q'):
            break

        plt.close() 

