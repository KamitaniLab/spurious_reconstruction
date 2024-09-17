# %%
import os
import pickle
from typing import Optional
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
# %%
def find_first_index_above_threshold(arr, threshold):

    """
    Returns the index of the first element in the array arr that exceeds the specified threshold.
    If no elements exceed the threshold, returns None.
    
    Parameters:
    arr (numpy.ndarray): The array to search.
    threshold (float): The threshold value.
    
    Returns:
    int or None: The index of the first element that exceeds the threshold, or None.
    """

    # Obtaining the boolean index of elements that exceed the threshold.ing
    above_threshold = arr > threshold
    
    # Obtaining the index of the first element that exceeds the threshold.
    indices = np.where(above_threshold)[0]
    
    if indices.size > 0:
        return indices[0]
    else:
        return -1
    
# %%

save_base_dir = './results/assets/fig09/'
os.makedirs(save_base_dir, exist_ok=True)

num_inner_clusters_list = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 
                                  
                                    ])
dim_list = [2, 4, 8, 16, 32 , 64 ,128, 256, 512, 1024, 2048, 4096, 8192 ]
# %%
# load results 
load_dir = './results/simulation_cluster_identification_dim/'
assert os.path.isdir(load_dir)

ratio_list = [0.1, 0.01, 0.5]

ratio = 0.1
with open(os.path.join(load_dir, f'cluster_identifiction_intra_inter_ratio{ratio}.pkl'), 'rb') as f:
    res_dict = pickle.load(f)
    
    
cond_dict = {
    "ratio 0.01":'cluster_identifiction_intra_inter_ratio0.01.pkl',
    "ratio 0.1": 'cluster_identifiction_intra_inter_ratio0.1.pkl',
    "ratio 0.5": 'cluster_identifiction_intra_inter_ratio0.5.pkl',
}

color_dict = {
    "ratio 0.01":0.8,
    "ratio 0.1": 0.5,
    "ratio 0.5":0.3,
}

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
thres = 0.5


for j, (cond_n, cond_f) in enumerate(cond_dict.items()):
    with open(os.path.join(load_dir, cond_f), 'rb') as f:
        res_dict = pickle.load(f)
        
        
    res_array = []
    test_array = []
    for i, dim in enumerate(dim_list):
        print(dim)
        #print(res_dict[dim]['oos_classification_accuracy'])
        test_array.append(res_dict[dim]['test_classification_accuracy'])
        res_array.append(res_dict[dim]['oos_classification_accuracy'])

    res_array = np.array(res_array)
    test_array = np.array(test_array)
    
    
    above_0_list = []
    above_50_list = []
    above_chance_list = []
    for i, arr in enumerate(res_array):
        print(arr)
        dim = dim_list[i]
        chance_performance = res_dict[dim]['chance_performances'][j]

        ind_50 = find_first_index_above_threshold(arr, thres )
        if ind_50 == -1: 
            cluster_50 = 0
        else:
            cluster_50 = num_inner_clusters_list[ind_50]
        above_50_list.append(cluster_50)


        ind_chance = find_first_index_above_threshold(arr, chance_performance)
        if ind_chance == -1: 
            cluster_chance = 0
        else:
            cluster_chance = num_inner_clusters_list[ind_chance]
        above_chance_list.append(cluster_chance)


        ind_0 = find_first_index_above_threshold(arr, 0)
        if ind_0 == -1: 
            cluster_0 = 0
        else:
            cluster_0 = num_inner_clusters_list[ind_0]
        above_0_list.append(cluster_0)

        
    ax.plot(dim_list[:-2], above_50_list[:-2], '-o', color=str(color_dict[cond_n]), label=cond_n)


ax.set_ylim([0, 750])
ax.set_xlabel('Dimension')
ax.set_ylabel('Number of cluster')
sns.despine()
ax.legend()
    
    
#fig.savefig(f'./results/assets/fig09/novel_cluster_identification_compare_thres{thres}.pdf')

#fig.savefig(f'./results/assets/fig09/novel_cluster_identification_compare_thres{thres}.png', dpi=600)
# %%

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
cmap = "rocket" #'gist_heat'
cmap1 = ax1.imshow(test_array, cmap = cmap)
cmap2 = ax2.imshow(res_array, cmap = cmap)


ax1.set_xticks(np.arange(len(num_inner_clusters_list
                           )))
ax1.set_xticklabels(num_inner_clusters_list, rotation=45, ha='right')
ax1.set_xlabel('Number of clusters')

ax1.set_yticks(np.arange(len(dim_list
                           )))
ax1.set_yticklabels(dim_list)
ax1.set_ylabel('Dimension')
 # Draw gridlines based on minor ticks
ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)

ax2.set_xticks(np.arange(len(num_inner_clusters_list
                           )))
ax2.set_xticklabels(num_inner_clusters_list, rotation=45, ha='right')
ax2.set_xlabel('Number of clusters')

ax2.set_yticks(np.arange(len(dim_list
                           )))
ax2.set_yticklabels(dim_list)
ax2.set_ylabel('Dimension')
ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)
# サブプロット間のスペースを調整
#plt.subplots_adjust(wspace=1.5) # wspaceを調整してサブプロット間の間隔を拡大



#fig.colorbar(cmap, cax=ax)
sns.despine()
#colorbar = plt.colorbar()
#fig.colorbar.set_ticks([0.0, 0.25, 0.5,  0.75, 1.0])
fig.colorbar(mappable=cmap1, ax=ax1, orientation='vertical', ticks=[0.0, 0.25, 0.5,  0.75, 1.0])
fig.colorbar(mappable=cmap2, ax=ax2, orientation='vertical', ticks=[0.0, 0.25, 0.5,  0.75, 1.0])

#fig.savefig(f'./results/assets/figS04/cluster_identification_confusion_martrix.pdf')

#fig.savefig('./results/assets/figS04/cluster_identification_confusion_martrix.png', dpi=600)


with open(os.path.join(load_dir, 'cluster_identifiction_intra_inter_ratio0.1.pkl'), 'rb') as f:
   res_dict = pickle.load(f)
fig, ax = plt.subplots(1,1)
#ax = axes[0]
ax.plot(num_inner_clusters_list,   res_dict[512]['test_classification_accuracy'], 'o-', label='Inside training', color=(26/255, 90/255, 46/255))
ax.plot(num_inner_clusters_list, res_dict[512]['oos_classification_accuracy'], 'o-', label='Outside training',color=(230/255, 88/255, 13/255))

ax.plot(num_inner_clusters_list, res_dict[512]['chance_performances'], '--', label='chance level',color='black')

#ax.vlines(x=64, ymin=0, ymax=1.0, colors='blue', ls='--')
#ax.vlines(x=128, ymin=0, ymax=1.0, colors='orange', ls='--')

ax.set_xscale('log')
#plt.yscale('log')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Cluster identification accuracy')
fig.legend()
sns.despine()
ax.set_title('Novel cluster identification')


#fig.savefig(os.path.join(save_base_dir, 'novel_cluster_identification.pdf')

