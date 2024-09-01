# %%
import os
import pickle
import numpy as np
import pandas as pd
import umap as up

import bdpy
from bdpy.dataform import Features

from umap_utils import extract_features, concatenate_features, norm_feat

# %%
def main():
    #network setting
    network = 'pytorch/brain_diffuser_versatile_diffusion/'
    feature_name = 'text_encoder'
    # %%
    # save dir setting
    save_base_dir = './results'
    assert os.path.isdir(save_base_dir)
    save_dir = os.path.join(save_base_dir, 'res_umap', network, feature_name)
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    # %%
    # feature path setting
    feat_dir_dict= {
        # NSD dataset [Allen et al, 2021]
        'nsd-train': f'/home/nu/data/contents_shared/NSD-stimuli/derivatives/features/{network}',
        'nsd-test':  f'/home/nu/data/contents_shared/NSD-stimuli/derivatives/features/{network}',
    }

    # stimulus id path setting
    stim_id_dict = {
        # NSD dataset [Allen et al, 2021]
        'nsd-train': f'/home/nu/data/contents_shared/NSD-stimuli/derivatives/nsd-analysis/nsd_stimulus_category_id_sub01_unique.csv',
        'nsd-test':  f'/home/nu/data/contents_shared/NSD-stimuli/derivatives/nsd-analysis/nsd_stimulus_category_id_sub01_shared1000.csv',
    }

    # Extract features using the new function
    features_dict = extract_features(feat_dir_dict, stim_id_dict, feature_name)

    # %%
    feats = concatenate_features(features_dict)

    # %%
    # perform umap
    # perform umap for clustering 
    # Refring to "UMAP enhanced clustering"
    # https://umap-learn.readthedocs.io/en/latest/clustering.html
    umap = up.UMAP(metric='cosine',
                        n_neighbors=30,
                        min_dist=0.0,
                        n_components=2,
                        random_state=42)
    # %%
    # normalize features
    feats_norm, m, s = norm_feat(feats)
    # perform umap
    embeddings = umap.fit_transform(feats_norm)
    np.save(f'{save_dir}/nsd_embedding_norm.npy', embeddings)
    print(f'Finish analysis saved at {save_dir}')


    if not os.path.isfile(f'{save_dir}/nsd_mean.npy'):
        np.save(f'{save_dir}/nsd_mean.npy', m)
    if not os.path.isfile(f'{save_dir}/nsd_norm.npy'):
        np.save(f'{save_dir}/nsd_norm.npy', s)

    # save umap 
    umap_name = 'nsd_umap.pkl'
    if not os.path.isfile(os.path.join(save_dir, umap_name)):
        pickle.dump(umap, open(os.path.join(save_dir, umap_name), 'wb'))

if __name__ == "__main__":
    main()