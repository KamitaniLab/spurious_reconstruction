# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%

load_base_dir = './results/res_umap'
assert os.path.isdir(load_base_dir)
save_base_dir = './results/assets/fig04'
network = 'pytorch/brain_diffuser_versatile_diffusion'
feat_name ='text_encoder' # 'vision_encoder'

#file_name = 'nsd_embedding_norm_default_param.npy'
prefix_list = ['nsd', 'deeprecon', 'both']
prefix = prefix_list[0]
file_name = f'{prefix}_embedding_norm.npy'


embeddings = np.load(os.path.join(load_base_dir, network, feat_name ,file_name))
num_train = 8859 

# Supplementary 
ss = 16

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
#ax.scatter(nsd_umap[:,0], nsd_umap[:,1], c=kmeans.labels_, cmap = 'tab20' , alpha=0.1)

ax.scatter(embeddings[:num_train][:,0],embeddings[:num_train][:, 1],  s=ss,linewidths=0.0,edgecolors='black', c="0.7", alpha=1.0, label='Trainning samples')

#ax.scatter(nsd_umap[:num_train][:,0],nsd_umap[:num_train][:, 1],  s=ss,linewidths=0.0,edgecolors='black', c="#3EB370", alpha=1.0, label='Trainning samples')

ax.scatter(embeddings[num_train:][:,0],embeddings[num_train:][:,1], s=ss-12,  linewidths=0.0, edgecolors='black', color=[(230/255, 88/255, 13/255)], alpha=1.0, label='Test')


ax.set_xticklabels([])
ax.set_yticklabels([])

plt.legend()

save_dir = save_base_dir
os.makedirs(save_dir, exist_ok=True)

#fig.savefig(os.path.join(save_base_dir, file_name.replace("npy", "pdf")))

#fig.savefig(os.path.join(save_base_dir, file_name.replace("npy", "png")), dpi=600)

# %%
