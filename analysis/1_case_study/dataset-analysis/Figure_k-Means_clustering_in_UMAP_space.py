# %%
import os
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import cluster
import pandas as pd
import random
import matplotlib
import random

load_base_dir = './results/res_umap'
assert os.path.isdir(load_base_dir)
save_base_dir = './results/assets/figS03'
network = 'pytorch/brain_diffuser_versatile_diffusion'
feat_name ='text_encoder' # 'vision_encoder'

#file_name = 'nsd_embedding_norm_default_param.npy'
prefix = 'nsd' #prefix_list[0]
file_name = f'{prefix}_embedding_norm.npy'

embeddings = np.load(os.path.join(load_base_dir, network, feat_name ,file_name))

# We observed 40 clusters in Umap space
cluster_num = 40
kmeans = cluster.KMeans(n_clusters=cluster_num, random_state=0, n_init="auto")
kmeans.fit(embeddings)


#heler function 
def choose_colors(num_colors):
  # matplotlib.colors.CSS4_COLORS 
  tmp = list(matplotlib.colors.CSS4_COLORS.values())
  random.seed(42)
  # shuffleing
  random.shuffle(tmp)
  label2color = tmp[:num_colors]
  return label2color

color_code = choose_colors(40)
hfont = {'fontname':'Helvetica'}

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
#ax.scatter(nsd_umap[:,0], nsd_umap[:,1], c=kmeans.labels_, cmap = 'tab20' , alpha=0.1)

for i in range(len(np.unique(kmeans.labels_))):
    ax.scatter(embeddings[kmeans.labels_==i,0], embeddings[kmeans.labels_==i,1], 
              c=color_code[i], label=i) 

for i in range(len(np.unique(kmeans.labels_))):
    x,y =np.median(embeddings[kmeans.labels_==i], 0)
    ax.scatter(x,y,  marker= f'${i+1}$', color='black', s=72)
#ax.set_title('k Means clustering (k=40)')
#ax.set_xlim(-10, 25)
#ax.set_ylim(-20, 20)
ax.set_xlim(-15, 20)
ax.set_ylim(-15, 30)


save_dir = save_base_dir
os.makedirs(save_dir, exist_ok=True)
#fig.savefig(os.path.join(save_base_dir, 'kmeans_cluster.pdf'))


# %%
