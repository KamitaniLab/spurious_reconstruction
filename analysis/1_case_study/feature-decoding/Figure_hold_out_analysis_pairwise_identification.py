
# %%
import argparse
import os
import sys
from natsort import natsorted
import bdpy
from bdpy.fig import makeplots
import numpy as np
import pandas as pd
import yaml
# %%
save_base_dir = './results/assets/fig06/pairwise_identification'
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
        analysis_name,
        conf['network']
    )

        
    perf_filename = os.path.join(dec_base_path,decoded_feature_test_dir, 'accuracy.pkl.gz')
    perf_cv_filename = os.path.join(dec_base_path, decoded_feature_test_dir,  'accuracy_fold.pkl.gz')
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
           # x='fold',  x_list=[cv_fold],
        x='condition',  x_list=condition_list,
           y='identification accuracy',
             subplot='layer', subplot_list=features,

            figure='subject',
            plot_type='bar',
            horizontal=True,
            reverse_x=True,
            title=f'{cv_fold}',
            style='seaborn-bright',

            y_lim=[0.4, 1],
        plot_size=(6,2),
        chance_level=0.5,
        fontsize=14, tick_fontsize=8,
            chance_level_style={'color': 'gray', 'linewidth': 1},
        
        )
    #assert 1== 0
    figs[0].savefig(os.path.join(save_base_dir, f'pw_iden_bar_{cv_fold}.pdf'))
# %%
