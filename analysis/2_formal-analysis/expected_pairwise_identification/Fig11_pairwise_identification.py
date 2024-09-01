# %%

import os
import numpy as np
import matplotlib.pyplot as plt
# %%


def calculate_pairwise_identification(k:int, q:float, pef_when_cand_is_same:float=0.5):
    """
    Calculate the pairwise identification accuracy 
    when test set comprises k categories and 
    all test samples are equally distributed across each category.
    This calculation is further assumed that he number of test samples is large enough.

    Parameters
    ----------
    k : int
        The number of categories
    q : float
        The accuracy of identifying a true samples correctly when the candidate is different category.
    pef_when_cand_is_same : float, optional
        The accuracy of identifying a true samples correctly when the candidate is the same category, by default 0.5.

    Returns
    -------
    float
        The pairwise identification accuracy
    
    """
   
    return 1/k *pef_when_cand_is_same + (1 - 1/k)*q

# %%
# Plotting
def plot_pairwise_identification_accuracy(accuracy_dict, ks):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    cmap = ['0.3', '0.5', '0.7']  # Color map for the plots
    for ii, (key, values) in enumerate(accuracy_dict.items()):
        ax.plot(ks, values, label=f'Identification accuracy across category: {key}', color=cmap[ii], marker='o', lw=1.5)
    ax.set_title('Pairwise identification accuracy')
    ax.set_xlabel('# of category in analysis set')
    ax.set_ylabel('Pairwise identification accuracy')

    ax.set_xticks(ks)
    ax.set_xticklabels(ks)

    ax.set_ylim([0.45, 1.0])
    ax.hlines(y=0.5, xmin=0, xmax=10, ls='--', color='black')
    ax.set_xlim([0.5, 10])
    ax.grid(color='gray', linewidth=0.6)
    plt.legend()
    plt.show()
    return fig



# %%

if __name__ == "__main__":
    # %%
    ks = np.arange(1, 11)
    qs = [1.0, 0.9, 0.8]


    accuracy_dict = {}
    for q in qs:
        pw_iden_perf = calculate_pairwise_identification(ks, q)
        accuracy_dict[q] = pw_iden_perf
        
    fig = plot_pairwise_identification_accuracy(accuracy_dict, ks)
# %%

    save_base_dir = './results/assets/'
    assert os.path.isdir(save_base_dir)
    save_dir = os.path.join(save_base_dir, "fig11")
    os.makedirs(save_dir, exist_ok=True)

    fig.savefig(os.path.join(save_dir, 'pairwise_identification_accuracy.pdf'))

    print('Done!')