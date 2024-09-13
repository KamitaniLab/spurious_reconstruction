'''DNN Feature decoding (corss-validation) training program'''


from __future__ import print_function

from itertools import product
import os
import shutil
from time import time
import warnings
import argparse

import bdpy
from bdpy.dataform import Features, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.ml.crossvalidation import make_cvindex_generator
from bdpy.util import makedir_ifnot
#from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Main #######################################################################

def average_cv_train_feat(
        fmri_data_files,
        features_dir,
        output_dir='./feature_decoding_cv',
        rois_list=None,
        num_voxel=None,
        label_key=None,
        cv_key='Run',
        cv_folds=None,
        cv_exclusive=None,
        features_list=None,
        feature_index_file=None,
        excluded_labels=[],
        alpha=100,
        chunk_axis=1
):
    '''Cross-validation feature decoding.

    Input:

    - fmri_data_files
    - features_dir

    Output:

    - output_dir

    Parameters:

    TBA

    Note:

    If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
    Note that Y[0] should be sample dimension.
    '''

    analysis_basename = os.path.splitext(os.path.basename(__file__))[0] + '-' + conf['__filename__']

    features_list = features_list[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(fmri_data_files.keys()))
    print('ROIs:            %s' % list(rois_list.keys()))
    print('Target features: %s' % features_dir)
    print('Layers:          %s' % features_list)
    print('CV:              %s' % cv_key)
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {
        sbj: bdpy.BData(dat_file[0])
        for sbj, dat_file in fmri_data_files.items()
    }

    if feature_index_file is not None:
        data_features = Features(os.path.join(features_dir), feature_index=feature_index_file)
    else:
        data_features = Features(os.path.join(features_dir))

    # Initialize directories -------------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        feature_index_save_file = os.path.join(output_dir, 'feature_index.mat')
        shutil.copy(feature_index_file, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Distributed computation setup ------------------------------------------
    distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
    distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')
    
    upd_cv_folds = []
    for cv_fold in cv_folds:
        if 'target' in cv_fold:
            test_cv = cv_fold['target']
        else:
            test_cv = cv_fold['test']
        upd_cv_fold = {'train': [test_cv[0] + 100],
                        'test': test_cv}
        upd_cv_folds.append(upd_cv_fold)
        

    for feat, sbj, roi in product(features_list, fmri_data_files, rois_list):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])

        # Cross-validation setup
        if cv_exclusive is not None:
            cv_exclusive_array = data_brain[sbj].select(cv_exclusive)
        else:
            cv_exclusive_array = None

        cv_index = make_cvindex_generator(
            data_brain[sbj].select(cv_key),
            folds=upd_cv_folds,
            exclusive=cv_exclusive_array
        )

        for icv, (train_index, test_index) in enumerate(cv_index):
            print('CV fold: {} ({} training; {} test)'.format(icv + 1, len(train_index), len(test_index)))

            # Setup
            # -----
            analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + str(icv + 1) + '-' + feat
            decoded_feature_dir = os.path.join(output_dir, feat, sbj, roi, 'cv-fold{}'.format(icv + 1), 'ave_decoded_features')
            os.makedirs(decoded_feature_dir, exist_ok=True)

            # Preparing data
            # --------------
            print('Preparing data')

            start_time = time()

            # Brain data
            #x = data_brain[sbj].select(rois_list[roi])       # Brain data
            x_labels = data_brain[sbj].get_label(label_key)  # Labels
            x_train_labels = np.array(x_labels)[train_index]
            

            # Y index to sort Y by X (matching samples)
            y_labels_unique = np.unique(x_train_labels)
            y_train_unique = data_features.get(feat, label=y_labels_unique)  # Target DNN features
            y_train_ave = np.mean(y_train_unique, 0, keepdims=True)
            # Save file name
            save_file = os.path.join(decoded_feature_dir, f'cv_fold{icv+1}.mat')

            # Save
            save_array(save_file, y_train_ave, key='feat', dtype=np.float32, sparse=False)
            
            
            # # Brain data
            # #x = data_brain[sbj].select(rois_list[roi])       # Brain data
            # x_test_labels = np.array(x_labels)[test_index]
            

            # # Y index to sort Y by X (matching samples)
            # y_labels_unique_test = np.unique(x_test_labels)
            # y_test_unique = data_features.get(feat, label=y_labels_unique_test)  # Target DNN features
            # y_test_ave = np.mean(y_test_unique, 0, keepdims=True)
            # # Save file name
            # save_file = os.path.join(decoded_feature_dir, f'test_cv_fold{icv+1}.mat')
            # # Save
            # save_array(save_file, y_test_ave, key='feat', dtype=np.float32, sparse=False)
        
            print('Elapsed time (data preparation): %f' % (time() - start_time))



    print('%s finished.' % analysis_basename)

    return output_dir


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

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    if 'analysis name' in conf:
        feature_decoders_dir = os.path.join(conf['feature decoder dir'], 'foldwise_ave_feature', conf['analysis name'], conf['network'])
    else:
        feature_decoders_dir = os.path.join(conf['feature decoder dir'],  'foldwise_ave_feature', conf['network'])

    if 'feature index file' in conf:
        feature_index_file = os.path.join(
            conf['training feature dir'][0],
            conf['network'],
            conf['feature index file']
        )
    else:
        feature_index_file = None

    if 'exclude test label' in conf:
        excluded_labels = conf['exclude test label']
    else:
        excluded_labels = []

    if 'cv folds' in conf:
        cv_folds = conf['cv folds']
    else:
        cv_folds = None

    if 'cv exclusive key' in conf:
        cv_exclusive = conf['cv exclusive key']
    else:
        cv_exclusive = None

    average_cv_train_feat(
        conf['fmri'],
        os.path.join(
            conf['feature dir'][0],
            conf['network']
        ),
        output_dir=feature_decoders_dir,
        rois_list=conf['rois'],
        num_voxel=conf['rois voxel num'],
        label_key=conf['label key'],
        cv_key=conf['cv key'],
        cv_folds=cv_folds,
        cv_exclusive=cv_exclusive,
        features_list=conf['layers'],
        feature_index_file=feature_index_file,
        excluded_labels=excluded_labels,
        alpha=conf['alpha'],
        chunk_axis=conf['chunk axis']
    )
