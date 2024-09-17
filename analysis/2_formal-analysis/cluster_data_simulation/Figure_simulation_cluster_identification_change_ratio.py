# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# %%
save_base_dir = './results/assets/figS04'
os.makedirs(save_base_dir, exist_ok=True)   

num_inner_clusters_list = np.array([32,64, 128, 256, 1024
                                        ])
show_inner_cluster_list = [32, 64, 128, 1024]

dim = 512
inner_intra_ratio_list = [0.01, 0.1, 0.5, 1.0, 5.0, 10, 50, 100]

# load results 
load_dir = './results/simulation_cluster_identification_ratio/'
    

color_dict = {
    "alpha: 10": 0.5,
}

cond_f = "cluster_identifiction_change_intra_inter_ratio.pkl"

# %%
with open(os.path.join(load_dir, cond_f), 'rb') as f:
    res_dict = pickle.load(f)
    
    
res_array = []
test_array = []
for i, ratio in enumerate(inner_intra_ratio_list):
    print(ratio)
    #print(res_dict[dim]['oos_classification_accuracy'])
    test_array.append(res_dict[ratio]['test_classification_accuracy'])
    res_array.append(res_dict[ratio]['oos_classification_accuracy'])

res_array = np.array(res_array)
test_array = np.array(test_array)

fig = plt.figure(figsize=(12, 8))

for j, show_cluster_num in enumerate(show_inner_cluster_list):
    #　num_inner_clusters_listから j に対応する 要素のindexを取得
    index = np.where(num_inner_clusters_list == show_cluster_num)[0][0]
    
    in_acc = test_array[:, index]
    oos_acc = res_array[:, index]
    
    ax = fig.add_subplot(2, 2, j+1)
    ax.plot(inner_intra_ratio_list, in_acc, 'o-', label='In-distribution', color=(26/255, 90/255, 46/255))
    ax.plot(inner_intra_ratio_list, oos_acc, 'o-', label='Out-of-distribution', color=(230/255, 88/255, 13/255))
    ax.set_xscale('log')
    ax.set_title(f'# training clusters = {show_cluster_num}')
    
#fig.savefig(os.path.join(save_base_dir, "/novel_cluster_identification_change_ratio.pdf")
