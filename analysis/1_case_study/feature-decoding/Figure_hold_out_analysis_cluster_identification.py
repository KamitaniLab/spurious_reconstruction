
# %%
import argparse
import os
import sys
from natsort import natsorted
from itertools import product
import bdpy
from bdpy.fig import makeplots
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix
# %%
save_base_dir = './results/assets/fig06/cluster_identification'
os.makedirs(save_base_dir, exist_ok=True)
featdec_path_dict = {
    
    
    'Hold_out' :'./analysis/1_case_study/feature-decoding/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml',
   'Naive' : './analysis/1_case_study/feature-decoding/config/umap_space_naive_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml',
    
   
    }


eval_list = []
eval_cv_list = []
dec_base_path = './'
for cond_name, perf_path in featdec_path_dict.items():
    print(cond_name)
    conf_file = os.path.join(perf_path)
    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)
        
    

        
    if 'analysis name' in conf:
        analysis_name = conf['analysis name']
    else:
        analysis_name = ''
        
    decoded_feature_test_dir = os.path.join(
        conf['decoded feature dir'],
        analysis_name +'_scaled',
        #'decoded_features',
        conf['network']
    )

        
    #perf_filename = os.path.join(dec_base_path,decoded_feature_test_dir, 'accuracy.pkl.gz')
    #perf_cv_filename = os.path.join(dec_base_path, decoded_feature_test_dir,  'accuracy_fold.pkl.gz')
    perf_filename = os.path.join(dec_base_path,decoded_feature_test_dir, 'sim_cluster_accuracy_classification.pkl.gz')
    perf_cv_filename = os.path.join(dec_base_path, decoded_feature_test_dir,  'sim_cluster_accuracy_fold_classification.pkl.gz')
    
    perf_df = pd.read_pickle(perf_filename)
    perf_cv_df = pd.read_pickle(perf_cv_filename)
    
    perf_df['condition'] = cond_name.replace('\n', '_')
    perf_cv_df['condition'] = cond_name.replace('\n', '_')
    print('Loaded {}'.format(perf_filename))

    print(perf_df)


    eval_list.append(perf_df)
    eval_cv_list.append(perf_cv_df)

# %%
tmp_df = pd.concat(eval_list)
tmp_cv_df = pd.concat(eval_cv_list)
fold_list = natsorted(np.unique(tmp_cv_df['fold']))
# %%
condition_list = ['Hold_out', 'Naive']
rois = conf['rois']
 
layers = conf['layers']
features = conf['layers']

# %%
for cv_fold in fold_list:
    print(cv_fold)
    cv_df = tmp_cv_df[tmp_cv_df['fold'] == cv_fold]
    
    figs = makeplots(
            cv_df,
            x='condition',  x_list=condition_list, 
             y='classification_performance',
             subplot='layer', subplot_list=features,

           # group ='condition',  group_list=condition_list[::-1], 
            figure='subject',
            plot_type='bar',
            horizontal=True,
            reverse_x=True,
           # x_label='Condition', y_label='Similar cluster identification accuracy',
            title=f'{cv_fold}',
            style='seaborn-bright',

            y_lim=[0, 1],
        plot_size=(6,2),
        chance_level=1/(39+1),
        #fontsize=24, tick_fontsize=24,
         fontsize=14, tick_fontsize=8,
            chance_level_style={'color': 'gray', 'linewidth': 1}
        )
    
    #assert 1== 0
    #figs[0].savefig(os.path.join(save_base_dir, f'cluster_iden_bar_{cv_fold}.pdf'))
# %%


save_base_dir = './results/assets/figS03/'
os.makedirs(save_base_dir, exist_ok=True)
for feat, condition in product(features, condition_list):
    test_df = tmp_df.query(f"layer=='{feat}' & condition== '{condition}'")
    
    true_y = test_df['labeled_cluster_ID'].values[0]
    pred_y = test_df['classified_cluster_ID'].values[0]
    
    cf =confusion_matrix(y_true = true_y, y_pred=pred_y)
    pcf = (cf.T/cf.sum(1)).T
    
    
    fig, ax =  plt.subplots(figsize=(2.5*10,2.5*10))
    ax.matshow(pcf, cmap=plt.cm.Blues, alpha=0.3)
    # Set ticks to be at every cell boundary
    ax.set_xticks(np.arange(-.5, 40, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 40, 1), minor=True)
    
    # Draw gridlines based on minor ticks
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)

    for i in range(pcf.shape[0]):
        for j in range(pcf.shape[1]):
            val = pcf[i,j]

            if val!=0:
                ax.text(x=j, y=i, s=f'{val:.02}', va='center', ha='center')


    #plt.xticks([0,4, 9, 14, 19, 24, 29, 34 ,39], [1,5,10,15, 20, 25, 30,35, 40])
    #plt.yticks([0,4, 9, 14, 19, 24, 29, 34 ,39], [1,5,10,15, 20, 25, 30,35, 40])
    #assert 1== 0

    
    ax.set_xticks(np.arange(40) )
    ax.set_xticklabels(np.arange(1,41))
    ax.set_yticks(np.arange(40) )
    ax.set_yticklabels(np.arange(1,41))
    #ax.grid(True, which='both', color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
    #ax,.set_.yticks(np.arange(40), np.arange(1,41))
    #plt.yticks([0,4, 9, 14, 19, 24, 29, 34 ,39], [1,5,10,15, 20, 25, 30,35, 40])
    plt.xlabel('Predicted cluster ID', size=24)
    plt.ylabel('True cluster ID', size=24)
    plt.title(f'Decoding Layer: {feat}  Condition: {condition}', size=24)
    
    #plt.show()
    
    plt.savefig(os.path.join(save_base_dir, f'confusion_probability_matrix_{feat}_{condition}.pdf'))
# %%
