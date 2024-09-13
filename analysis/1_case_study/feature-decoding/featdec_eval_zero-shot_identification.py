import argparse
import os
from glob import glob
import copy
import torch
import numpy as np
import pandas as pd
import yaml


from itertools import product
import os
import re

import bdpy

from bdpy.dataform import Features, DecodedFeatures

import numpy as np
import pandas as pd
import scipy
import yaml
from bdpy.ml.crossvalidation import make_cvindex_generator
from bdpy.evals.metrics import pairwise_identification
import argparse
import os
from glob import glob
from scipy.spatial.distance import cdist


def compute_identification_accuracy(pred, test, cand, metric ='correlation',top_k=1):
    """
    Compute the C+1-wise identification accuracy.
    Parameters
    ----------
    pred : np.ndarray
        The predicted output having shape (num_test, output_dim)
        test : np.ndarray
        The target output having shape (num_test, output_dim)
    cand : np.ndarray
        The candidate output having shape (num_candidates, output_dim)
    metric: str, optional
        The distance metric to use, by default 'correlation'
    top_k : int, optional
    
    """
    # compute the distance between the predicted and the test
    cand = np.concatenate([test, cand])
    dist = cdist(pred, cand, metric)
    # get the top k index
    top_k_index = np.argsort(dist, axis=1)[:, :top_k]
    # if the test is in the top k index, then it is correctly identified
    accuracy = np.any(top_k_index == np.arange(len(test))[:, None], axis=1)
    
    return accuracy


# Entry point ################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
    )
    
    args = parser.parse_args()
    
    save_base_dir = "./results/zero-shot_identification"
    os.makedirs(save_base_dir, exist_ok=True)

    conf_file = args.conf
    with open(conf_file, 'r') as f:
        conf_original = yaml.safe_load(f)
    device = args.device

    original_feature_path =os.path.join(
                conf_original['decoded feature dir'],
            conf_original['analysis name'],
                "decoded_features",
                conf_original['network']
            )
    features_list=conf_original['layers'][::-1]
    roi = list(conf_original['rois'].keys())[0]
    # load decoded feature
    pred_features = DecodedFeatures(os.path.join(original_feature_path), squeeze=False)
    fmri_data_files = conf_original['training fmri']
    data_brain = {sbj: bdpy.BData(dat_file[0])
                  for sbj, dat_file in fmri_data_files.items()}
    label_key = conf_original['label key']
    train_features_dir = os.path.join(
            conf_original['training feature dir'][0],
            conf_original['network']
        )
    data_train_features = Features(os.path.join(train_features_dir))
    
    test_features_dir = os.path.join(
            conf_original['test feature dir'][0],
            conf_original['network']
        )
    data_test_features = Features(os.path.join(test_features_dir))
    
    
    #x_labels = data_brain["nsd-01"].get_label(label_key)  # Labels
    x_labels = data_brain["nsd-01"].get_label(label_key)  # Labels
    y_train_labels = np.unique(x_labels)
    test_labels = np.unique(pred_features.labels)

    for feat in features_list:
        save_dir = os.path.join(save_base_dir,conf_original['network'], feat)
        os.makedirs(save_dir, exist_ok=True)
        
        
        
        perf_df_fold = pd.DataFrame(columns=[
            'image_id', 
            'vs_training_samples', 'vs_training_samples_top10', 'vs_training_samples_top50',
            'vs_traininig_cluster', 'pairwise_training_samples'
        ])
      
            
        y_train = data_train_features.get(feat, label=np.unique(y_train_labels))  # Target DNN features
            
        cand_feat_flat = y_train.reshape(y_train.shape[0], -1)
            
            
        y_test = data_test_features.get(feat, label=test_labels)
        #y_test = np.array([data_test_features.get(layer=feat, label=test_label) for test_label in test_labels])
        # feature scaling is needed
        test_feat_flat = y_test.reshape(y_test.shape[0], -1)
        #assert 1== 0
        pred_feats = np.array([pred_features.get(layer=feat, subject='nsd-01', roi=roi, label=test_label) for test_label in test_labels])

            
        pred_feat_flat = pred_feats.reshape(pred_feats.shape[0],-1)
        
        vs_training_samples = [compute_identification_accuracy(pred_feat_flat[i][np.newaxis], test_feat_flat[i][np.newaxis], cand_feat_flat)[0] for i in range(len(y_test))]
        


        pairwise = [pairwise_identification(pred_feat_flat[i][np.newaxis], np.concatenate([test_feat_flat[i][np.newaxis], cand_feat_flat])) for i in range(len(y_test))]
        
        print('novelty identification')
        print(np.mean(vs_training_samples))
  
    
        print('pairwise identification')
        print(np.mean(pairwise))
        
        # save_df_fold
        new_data = pd.DataFrame({
            'image_id': test_labels,
            'vs_training_samples': vs_training_samples,
            'pairwise_training_samples': pairwise,
            'num_of_candidate': len(cand_feat_flat)
        })
        perf_df_fold = pd.concat([perf_df_fold, new_data], ignore_index=True)
        
        # save as csv
        perf_df_fold.to_csv(os.path.join(save_dir, 'performance_correlation.csv'), index=False)
            
                
            
            

                
                
                
                

                
                


