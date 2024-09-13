# %%

import os
from glob import glob
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
# %%
save_base_dir = "./results/assets/fig05"
os.makedirs(save_base_dir, exist_ok=True)
# %%
total_test_samples = 982 #"subject 1 unique stimulus",

total_training_samples = 8859 #"subject 1 unique stimulus",

dataset = "NSD"
print(dataset)
# %%
network_dict = {
    "pytorch/brain_diffuser_versatile_diffusion": ["text_encoder", "vision_encoder"],
    "caffe/VGG_ILSVRC_19_layers": [ #"fc8", "fc6"
                                    "fc6",     #"conv5_4",
                                    "conv5_3", #"conv5_2", 
                                    "conv5_1", #"conv4_4",
                                    "conv4_3", #"conv4_2",
                                    "conv4_1", #"conv3_4",
                                    "conv3_3", #"conv3_2",
                                    #"conv3_1", "conv2_2", # 
                                    #"conv2_1", "conv1_2", 
                                    #"conv1_1", 
                                  ]
}


# %%
#load_base_dir = './data/NSD-stimuli/derivatives/zero-shot_sample_identification/NSD'
load_base_dir = "./results/zero-shot_identification"
# %%

for network, layer_list in network_dict.items():
    perf_df = pd.DataFrame(columns=[
                        'network',
                        'layer', 
                       "candidate_sample",
                        'test_sample', 
        ])



    performance_dict = {}
    res_list = []
    num_within_category_list = []
    for feat in layer_list:
        print(feat)
        load_dir = os.path.join(load_base_dir, network, feat)

        #load df 
        tmp_df = pd.read_csv(os.path.join(load_dir, 'performance_correlation.csv'))
        total_test_in_fold = len(tmp_df)
        generalize_test_in_fold = tmp_df['vs_training_samples'].astype(int)

        sample =  total_training_samples#8859
        num_within_category_list.append(sample)
        val = tmp_df['pairwise_training_samples']

        key= f'{network}-{feat}'
        #performance_dict
        res_list.append(generalize_test_in_fold)

        print(generalize_test_in_fold.sum())
        perf_df = pd.concat([
            perf_df, 
            pd.DataFrame.from_dict({
                        "network": [network.split('/')[-1]],
                        'layer':   feat, 
                        'novel_sample_identification': generalize_test_in_fold.sum(), 
                        "test_sample": total_test_in_fold,
                        'novel_sample_identification_accuracy': generalize_test_in_fold.sum() /len(val),
                        
                    }),
                    
        ], ignore_index=True)
    categories =  perf_df["layer"]
    correct_samples = perf_df["novel_sample_identification"]
    incorrect_samples = perf_df["test_sample"] - perf_df["novel_sample_identification"]
    
    
    fig, ax = plt.subplots()
    bar_width = 0.35
    net = network.split('/', )[-1]
    # bar graph of correct samples 
    rects1 = ax.barh(categories, correct_samples, bar_width, label='Correct', color='orange')

    # cumurative bar graph of uncorrect samples
    rects2 = ax.barh(categories, incorrect_samples, bar_width, left=correct_samples, label='Incorrect', color='0.6')

    ax.vlines(x = 982, ymin=-0.5, ymax=len(categories)-0.5, color='0.8', ls='--')

    # ラベルやタイトルの設定
    ax.set_ylabel('Features')
    ax.set_xlabel('Number of Test samples')
    ax.set_title('Zero shot sample indentification')

    #ax.legend()

    # グラフを表示
    plt.show()

    fig.savefig(os.path.join(save_base_dir, f'zero_shot_sample_identification_accuracy_{dataset}_{net}_cumerative_plot.pdf'), bbox_inches='tight')

                
# %%
