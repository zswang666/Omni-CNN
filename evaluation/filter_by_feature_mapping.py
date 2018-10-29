import numpy as np
import pdb

from filter_by_feature_mapping_utils import *

def filter_by_feature_mapping(query_feature, filtered_features, k, **kwargs):
    """
    Choose the index from the filtered_features that maps the top k best to the query_feature
    Input:
        query_feature: np.3darray [H,W,C] float; the feature of query image
        filtered_features: np.4darray [B,H,W,C] float; the feature of node candidates, there are B of them
        k: int; the number of top chosen nodes that maps best to the query feature
    Return:
        filtered_candidate: list of int; the index of filtered k candidate from B candidates, from most fit to less fit
    """
    kwargs.setdefault('dist_thresh', 15)
    kwargs.setdefault('min_pts', int(0.4*k))
    kwargs.setdefault('bandwidth', 1.)

    query_feature = np.expand_dims(query_feature, axis=0)
    packBatch = np.vstack([query_feature,filtered_features])
    distMatList, pairList, topKDistList = feature_matching(packBatch, k)
    fPairList, transList, densityList, fBoolList, consList = analyze_pairs(pairList,
                                                                           topKDistList,
                                                                           dist_thresh=kwargs['dist_thresh'],
                                                                           min_pts=kwargs['min_pts'],
                                                                           cons_thresh=5, # no use
                                                                           bandwidth=kwargs['bandwidth'])

    zz_list = []
    for density in densityList:
        if density is not None:
            _, _, zz = density
            zz /= zz.sum()
            zz_list.append(zz.max())
        else:
            zz_list.append(-99999)

    filtered_candidate = np.argsort(zz_list).tolist()[:k]

    return filtered_candidate
