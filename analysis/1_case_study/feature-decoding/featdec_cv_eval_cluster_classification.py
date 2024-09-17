'''Feature decoding evaluation.'''


import argparse
from itertools import product
import os
import re

from bdpy.dataform import Features, DecodedFeatures
from hdf5storage import loadmat
import numpy as np
import scipy
import pandas as pd
import scipy
import yaml

def perform_image_identification_multi_cand(pred, true, cand,metric='correlation',
                                  remove_nan=True, remove_nan_dist=True):
    
    lure_num = cand.shape[1]
    pred = pred.reshape(pred.shape[0], -1)
    true = true.reshape(true.shape[0], -1)
    cand = cand.reshape(cand.shape[0], lure_num, -1)
    cand_num = lure_num+1
    #res = 0
    res_list = []
    for d_ind in range(len(pred)):
       #res = 0
        p = pred[d_ind]
        c = true[d_ind]
        s = cand[d_ind]

        pc = np.corrcoef(p,c)[0,1]
        ps =  np.array([np.corrcoef(p,s[i])[0,1] for i in range(len(s))])
        #pw_iden = np.mean(pc > ps)
        
        if np.sum(pc > ps) == (cand_num -1):
            res=1
        else:
            res=0
        res_list.append(res)
    return np.array(res_list)#pw_iden


def perform_NN_classification(y_pred_feat, cand_feat):
    
    pred = y_pred_feat.reshape(y_pred_feat.shape[0], -1)
    cand = cand_feat.reshape(cand_feat.shape[0], -1)
    
    res_list = []
    for d_ind in range(len(pred)):
        p = pred[d_ind]
        
        ps =  np.array([np.corrcoef(p,cand[i])[0,1] for i in range(len(cand))])
        res = np.argmax(ps)
        res_list.append(res)
    
    return np.array(res_list)


# Main #######################################################################

def decfeat_cv_eval_sim_cluster_iden(
        decoded_feature_dir,
        cluster_ave_feature_dir, 
        true_feature_dir,
        output_file_pooled='./sim_cluster_identification_accuracy.pkl.gz',
        output_file_fold='./sim_cluster_identification_accuracy_fold.pkl.gz',
        subjects=None,
        rois=None,
        cand_num=39,
        features=None,
        feature_index_file=None,
        feature_decoder_dir=None,
        single_trial=False
):
    '''Evaluation of feature decoding.

    Input:

    - deocded_feature_dir
    - true_feature_dir

    Output:

    - output_file

    Parameters:

    TBA
    '''

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Decoded features: {}'.format(decoded_feature_dir))
    print('')
    print('True features (Test): {}'.format(true_feature_dir))
    print('')
    print('Layers: {}'.format(features))
    print('')
    if feature_index_file is not None:
        print('Feature index: {}'.format(feature_index_file))
        print('')

    # Loading data ###########################################################

    # True features
    if feature_index_file is not None:
        features_test = Features(true_feature_dir, feature_index=feature_index_file)
    else:
        features_test = Features(true_feature_dir)

    # Decoded features
    decoded_features = DecodedFeatures(decoded_feature_dir)

    cv_folds = decoded_features.folds

    # Evaluating decoding performances #######################################

    if os.path.exists(output_file_fold):
        print('Loading {}'.format(output_file_fold))
        perf_df_fold = pd.read_pickle(output_file_fold)
    else:
        print('Creating an empty dataframe')
        perf_df_fold = pd.DataFrame(columns=[
            'layer', 'subject', 'roi', 'fold',
            'profile correlation', 'labeled_cluster_ID', 'classified_cluster_ID', 'classification_performance'
        ])

    #true_labels = features_test.labels
    decoded_labels = decoded_features.labels
    fold_num = len(cv_folds)
    fold_list = ['cv-fold' + str(cv_index + 1) for cv_index in range(fold_num)]
    for layer in features:
        print('Layer: {}'.format(layer))
        #true_y = features_test.get_features(layer=layer,label=true_labels)
        true_y = features_test.get(layer=layer,label=decoded_labels)
        true_labels = decoded_labels
        for subject, roi in product(subjects, rois):
            load_dir = os.path.join(cluster_ave_feature_dir, layer, subject, roi)
            #load_file = os.path.join(load_dir, f'{layer}.mat')
            load_file_list = [os.path.join(load_dir, fold, 'ave_decoded_features', fold.replace('-', '_')+'.mat') for fold in fold_list]
            ave_feature_list = np.array([loadmat(load_file)['feat'] for load_file in load_file_list]).squeeze()#loadmat(load_file)

            #aa = loadmat(load_file)
            #ave_training_label = [label.split(' ')[0] for label in aa['ave_training_label']]
            #cdist_corr_df = pd.DataFrame(aa['cdist_corr'], index=aa['test_label'], columns=fold_list)
            #ave_feature_list = aa['foldwise_ave_feat']
            for j, fold in enumerate(fold_list):#cv_folds):
                print('Subject: {} - ROI: {} - Fold: {}'.format(subject, roi, fold))
                
                if len(perf_df_fold.query(
                        'layer == "{}" and subject == "{}" and roi == "{}" and fold == "{}"'.format(
                            layer, subject, roi, fold
                        )
                )) > 0:
                    print('Already done. Skipped.')
                    continue

                # Load features
                scaled_pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi, fold=fold)
                pred_labels = decoded_features.selected_label


                if single_trial:
                    pred_labels = [re.match('trial_\d*-(.*)', x).group(1) for x in pred_labels]

                if not np.array_equal(pred_labels, true_labels):
                    y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                    true_y_sorted = true_y[y_index]
                else:
                    true_y_sorted = true_y
                    
                    
                ## Select similar fold
                classified_res = perform_NN_classification(y_pred_feat = scaled_pred_y, cand_feat= ave_feature_list)
                true_ID = np.array([j] * len(scaled_pred_y))
                #print('Median classified cluster: {}'.format(scipy.stats.mode(classified_res)[0]))
                true_ID == classified_res
                classification_performance = (true_ID == classified_res) * 1
                classification_accuracy = np.sum(classification_performance)/len(true_ID)
                print('Classification accuracy: {}'.format(classification_accuracy ))
                
                perf_df_fold = pd.concat([perf_df_fold, pd.DataFrame({
                    'layer':   [layer],
                    'subject': [subject],
                    'roi':     [roi], 
                    'fold':    [fold],
                    'labeled_cluster_ID': [true_ID],
                    'classified_cluster_ID': [classified_res],
                    'classification_performance': [classification_performance]
                })], ignore_index=True)


    print(perf_df_fold)

    # Save the results (each fold)
    perf_df_fold.to_pickle(output_file_fold, compression='gzip')
    print('Saved {}'.format(output_file_fold))

    print('All done')

    # Pool accuracy
    perf_df_pooled = pd.DataFrame(columns=[
        'layer', 'subject', 'roi',
        'profile correlation', 'pattern correlation', 'labeled_cluster_ID', 'classified_cluster_ID',
        'classification_performance'
    ])

    for layer, subject, roi in product(features, subjects, rois):
        q = 'layer == "{}" and subject == "{}" and roi == "{}"'.format(layer, subject, roi)
        r = perf_df_fold.query(q)

        #r_prof_pooled = r['profile correlation'].mean()
        #r_patt_pooled = np.hstack(r['pattern correlation'])
        gt_id_pooled  = np.hstack(r['labeled_cluster_ID'])
        pred_id_pooled  = np.hstack(r['classified_cluster_ID'])
        classification_performance_pooled = np.hstack(r['classification_performance'])
        
        perf_df_pooled = pd.concat([perf_df_pooled, pd.DataFrame({
            'layer':   [layer],
            'subject': [subject],
            'roi':     [roi],
            'labeled_cluster_ID': [gt_id_pooled.flatten()],
            'classified_cluster_ID': [pred_id_pooled.flatten()],
            'classification_performance': [classification_performance_pooled.flatten()]
        })], ignore_index=True)

    print(perf_df_pooled)

    # Save the results (pooled)
    perf_df_pooled.to_pickle(output_file_pooled, compression='gzip')
    print('Saved {}'.format(output_file_pooled))

    print('All done')

    return output_file_pooled, output_file_fold


# Entry point ################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    parser.add_argument(
        '--cand_images',
        type=int,
        default=39,
        
    )
    args = parser.parse_args()

    conf_file = args.conf
    cand_num = args.cand_images
    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    if 'analysis name' in conf:
        analysis_name = conf['analysis name']
    else:
        analysis_name = ''
    # 
    scaled_decoded_feature_dir = os.path.join(
        conf['decoded feature dir'],
        analysis_name + '_scaled',
        conf['network']
    )
    
    cluster_ave_feature_dir = os.path.join(
        conf['decoded feature dir'],
        'foldwise_ave_feature',#'cdist_corr_with_trianing_images_umap_clustered_ave_feat', 
        analysis_name.replace('naive', 'holdout'),
        conf['network']
    )

    if 'feature index file' in conf:
        feature_index_file = os.path.join(conf['training feature dir'][0], conf['network'], conf['feature index file'])
    else:
        feature_index_file = None

    if 'test single trial' in conf:
        single_trial = conf['test single trial']
    else:
        single_trial = False

    save_base_dir = f'./results/cluster_identification/{analysis_name}'
    os.makedirs(save_base_dir, exist_ok=True)
    decfeat_cv_eval_sim_cluster_iden(
        scaled_decoded_feature_dir,
        cluster_ave_feature_dir, 
        os.path.join(conf['feature dir'][0], conf['network']),
        output_file_pooled=os.path.join(save_base_dir, f'sim_cluster_accuracy_classification.pkl.gz'),
        output_file_fold=os.path.join(save_base_dir, f'sim_cluster_accuracy_fold_classification.pkl.gz'),
        subjects=list(conf['fmri'].keys()),
        rois=list(conf['rois'].keys()),
        features=conf['layers'],
        feature_index_file=feature_index_file,
        feature_decoder_dir=os.path.join(
            conf['feature decoder dir'],
            analysis_name,
            conf['network']
        ),
        cand_num=cand_num,
        single_trial=single_trial
    )
