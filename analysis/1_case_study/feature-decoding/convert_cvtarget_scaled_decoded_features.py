# coding: utf-8

'''
Decoded feature scaling script.
Targetありのcvを行う場合専用
targetを除くtest decoded featureでmeanとstdを計算し、targetのtest decoded featureをscalingする
'''

import os
from itertools import product 
import argparse

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import hdf5storage

import bdpy
from bdpy.dataform import Features, DecodedFeatures
from bdpy.util import makedir_ifnot
from bdpy.dataform import load_array, save_array


def convert_save_cvtarget_scaled_feature(
        fmri_data_files,
        decoded_feature_dir,
        scaled_decoded_feature_dir,
        feature_decoder_dir,
        label_key=None,
        subjects_list=None,
        rois_list=None,
        features_list=None,
        cv_folds=None,
        cv_key=None,
):
    '''
    Scale the decoded feature regarding to true training feature std.
    On Cross validation.
    '''
    
    print("Original dec:", decoded_feature_dir)
    print("Save to:", scaled_decoded_feature_dir)

    # Load original decoded feautres
    dec_features = DecodedFeatures(decoded_feature_dir, squeeze=False)

    for sbj in subjects_list:
        print("Load brain data...")
        data_brain = bdpy.BData(fmri_data_files[sbj][0]) # とりあえず1ファイルだけ読み出しにしておく
        labels = np.array(data_brain.get_label(label_key))
        cv_labels = data_brain.get(cv_key).flatten()
        print("Done.")
        for roi, layer in product(rois_list, features_list):
            for cvi, cv in enumerate(cv_folds):
                fold = "cv-fold" + str(cvi + 1)
                print(sbj, roi, layer, fold)
                
                # Create the path for saving scaled decoded features
                results_dir_prediction = os.path.join(scaled_decoded_feature_dir, layer, sbj, roi, fold, 'decoded_features')
                if os.path.exists(results_dir_prediction):
                    print("Already exists:", results_dir_prediction)
                    continue;
                else:
                    os.makedirs(results_dir_prediction)
                    print("Convert to:", results_dir_prediction)
                
                # Select target labels
                cv_test = cv['test']
                cv_target = cv['target']
                #cv_target = cv['test']
                
                cv_target_index = np.any(np.vstack([cv_labels == cvt for cvt in cv_target]), axis=0)
                cv_target_labels = list(np.unique(labels[cv_target_index]))

                cv_test_index = np.any(np.vstack([cv_labels == cvt for cvt in cv_test]), axis=0)
                # arrayのboolean同士は演算できないので一旦intに直して，test分を除去する
                cv_test_index_for_scaling = np.array(cv_test_index.astype(np.int64) - cv_target_index.astype(np.int64), dtype=np.bool_) 
                cv_test_labels_for_scaling = list(np.unique(labels[cv_test_index_for_scaling]))

                # Load features
                # DecodedFeaturesクラスはlabelのリストに対応していないため，labelごとに読み出し，stackで連結
                dec_feat_target = np.stack([dec_features.get(layer=layer, subject=sbj, roi=roi, fold=fold, label=tlbl) for tlbl in cv_target_labels])  
                dec_feat_test_for_scaling = np.stack([dec_features.get(layer=layer, subject=sbj, roi=roi, fold=fold, label=tlbl) for tlbl in cv_test_labels_for_scaling])
                dec_mean = np.mean(dec_feat_test_for_scaling, axis=0) # scaling用のtest setだけ使うこと
                dec_std = np.std(dec_feat_test_for_scaling, axis=0, ddof=1)
                train_norm_param_dir = os.path.join(
                        feature_decoder_dir,
                        layer, sbj, roi, fold,
                        'model'
                    )
                true_train_mean = hdf5storage.loadmat(os.path.join(train_norm_param_dir, 'y_mean.mat'))['y_mean']
                true_train_std = hdf5storage.loadmat(os.path.join(train_norm_param_dir, 'y_norm.mat'))['y_norm']
                print("true_train_mean.shape", true_train_mean.shape)
                print("true_train_std.shape", true_train_std.shape)
                print("dec_mean.shape", dec_mean.shape)
                print("dec_std.shape", dec_std.shape)

                # Scaling
                scaled_dec = (dec_feat_target - dec_mean) / dec_std * true_train_std + true_train_mean
                print("scaled_dec.shape", scaled_dec.shape)
                    
                # Save scaled features
                for i, label in enumerate(cv_target_labels):
                    # Predicted features
                    # 上をvstackではなくstackで実装したため，ここでは np.array([scaled_dc[i]]) とする必要がなくなった
                    feat = scaled_dec[i]  # To make feat shape 1 x M x N x ...
                    if i == 0:
                        print("feat.shape", feat.shape)
                    
                    # Save file name
                    save_file = os.path.join(results_dir_prediction, '%s.mat' % label)
                    
                    # Save
                    if not os.path.exists(save_file):
                        print("Save to:", save_file)
                        save_array(save_file, feat, key='feat', dtype=np.float32, sparse=False)


# Entry point ################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    args = parser.parse_args()

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    if 'analysis name' in conf:
        analysis_name = conf['analysis name']
    else:
        analysis_name = ""

    feature_decoder_dir=os.path.join(
        conf['feature decoder dir'],
        analysis_name,
        conf['network']
    )

    decoded_feature_dir = os.path.join(
        conf['decoded feature dir'],
        analysis_name,
        conf['network']
    )
    
    scaled_decoded_feature_dir = os.path.join(
        conf['decoded feature dir'],
        analysis_name + '_scaled',
        conf['network']
    )
    

    convert_save_cvtarget_scaled_feature(
        conf['fmri'],
        decoded_feature_dir,
        scaled_decoded_feature_dir,
        feature_decoder_dir,
        label_key=conf['label key'],
        subjects_list=list(conf['fmri'].keys()),
        rois_list=list(conf['rois'].keys()),
        features_list=conf['layers'],
        cv_folds=conf['cv folds'],
        cv_key=conf['cv key']
    )
