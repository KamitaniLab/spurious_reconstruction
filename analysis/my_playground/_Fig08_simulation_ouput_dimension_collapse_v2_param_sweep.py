# This script simulates the phenomenon of output dimension collapse (Figure 8).
# Output dimension collapse occurs when the dimensionality of the output space
# is effectively reduced due to the limitations or characteristics of the model or data.
# This simulation helps in understanding how different models handle or are affected by
# reduced dimensionality in output spaces, which is crucial for tasks like dimensionality reduction,
# feature extraction, and more complex scenarios like anomaly detection or data compression.

# %%
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from simulation_utils import  Simulator, rotate_array
from simulation_cluster_identificaiton import SimpleClusterDataFactory

from itertools import product
from scipy.stats import gaussian_kde

# %%
# Data class
class ClusterProjectionDataFactory(SimpleClusterDataFactory):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_inner_clusters: int,
        num_outer_clusters: int,
        noise_var: float = 0.5,
        cluster_var: float = 0.01,
        inter_cluster_var: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the data factory.

        RondomProjectionDataFactory generates input and output data pairs.
        The data generation process is as follows:

        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        num_inner_clusters : int
            The number of clusters for the inner data
        num_outer_clusters : int
            The number of clusters for the outer data
        noise_var : float, optional
            The variance of the observational noise, by default 0.5
        cluster_var : float, optional
            The variance of the cluster (cluster radius), by default 0.01
        inter_cluster_var": float, optional
            The variance of the inter-cluster distance, by default 1.0
        seed : Optional[int], optional
            The seed for the random number generator, by default None
        """
        super().__init__(input_dim, output_dim,
        num_inner_clusters, num_outer_clusters,
        noise_var,
        cluster_var, inter_cluster_var, seed)


    def generate_outer(self, num: int, scale_value: int =1): # -> tuple[np.ndarray, np.ndarray]:
        """
        Generate outer data.

        Parameters
        ----------
        num : int
            The number of data to generate
        scale_value: float
            The scale values of random gaussian distribution
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The generated data. The first element is the input data and the
            second element is the output data.
        """
        assignments = self._rng.choice(self.num_outer_clusters, size=num, replace=True)
        y = np.mean(self.inner_cluster_centers,0) + self._rng.normal(
            scale=np.sqrt(self.inter_cluster_var), size=(num, self.output_dim)
        )
        # Way 1: generate gaussian at Brain space matching the scale of 
        #x = np.mean(self.inner_cluster_centers,0)@ self.W + self._rng.normal(
        #    scale=(np.sqrt(self.cluster_var + self.noise_var)) * scale_value, size=(num, self.input_dim)
        #)
        #x = y @ self.W + self._rng.normal(
        #    scale=np.sqrt(self.noise_var) * scale_value, size=(num, self.input_dim)
        #)
        #Way 2: generate gaussian at Brain space matching the scale of cluster features 
        #x = self._rng.normal(
        #    scale=np.sqrt(self.inter_cluster_var + self.noise_var) * scale_value, size=(num, self.input_dim)
        #)
        
        
        #Way 3: generate gaussian at Brain space matching the scale of cluster features 
        y_rand = self._rng.normal(
            scale=np.sqrt(self.inter_cluster_var+self.cluster_var) * scale_value , size=(num, self.input_dim)
        )
        #y_rand = self._rng.uniform(-np.sqrt(self.inter_cluster_var+self.cluster_var) * scale_value, np.sqrt(self.inter_cluster_var+self.cluster_var) * scale_value, size=(num, self.input_dim)
        #    
        #)
        x = y_rand @ self.W + self._rng.normal(
            scale=np.sqrt(self.noise_var), size=(num, self.input_dim)
        )

        return x, y,assignments
    
#Figure setting

class PlotSettings:
    def __init__(self):
        self.nrow = 2
        self.ncol = 3
        self.size = 9
        self.xx = np.arange(-200, 200)
        self.colors = plt.cm.Blues
        self.c = self.colors(128)
        self.num_select = 10000#800
        self.num_select_train = 50
        self.alpha = 1.0
        self.angle = 32
        self.y_axis = 3
        self.x_axis = 1
        np.random.seed(1)
    
def plot_brain_and_latent_spaces(simulator, plot_settings, embed=True, with_true_oos=False):
    class DataSelection:
        def __init__(self, simulator):
            self.simulator = simulator
            self.x_train = simulator.X_train
            self.x_test = simulator.X_oos
            self.X_test_ind = np.random.choice(np.arange(len(simulator.X_oos)), plot_settings.num_select)
            self.X_test = self.x_test[self.X_test_ind]
            self.X_train_ind = np.random.choice(np.arange(len(simulator.X_train)), plot_settings.num_select_train)
            self.X_train = self.x_train[self.X_train_ind]
        
    data_selection = DataSelection(simulator)

    if embed:
        y_train = simulator.emb_Y_train
        y_test = simulator.emb_Y_test
        y_oos = simulator.emb_Y_oos
        y_pred = simulator.emb_Y_oos_dec
    else:
        y_train = simulator.Y_train
        y_test = simulator.Y_test
        y_oos = simulator.Y_oos
        y_pred = simulator.y_oos_pred_dec

    # 図の作成
    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(121)
    
    data_test  = [data_selection.X_test[...,plot_settings.x_axis], data_selection.X_test[...,plot_settings.y_axis]]
    data_train = [data_selection.X_train[...,plot_settings.x_axis], data_selection.X_train[...,plot_settings.y_axis]]
    ax.scatter(data_test[0],data_test[1],
               c='0.8', 
               label="Random input", 
               s=plot_settings.size, alpha=plot_settings.alpha, linewidths=0.0, edgecolors='white')

    ax.scatter(data_train[0], data_train[1],
               c=[(26/255, 90/255, 46/255)], 
               label="Training input", s=plot_settings.size, 
               alpha=plot_settings.alpha, linewidths=0.0, edgecolors='white')

    d_min = np.min(np.concatenate([data_test, data_train],axis=1))
    d_max = np.max(np.concatenate([data_test, data_train],axis=1))
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(d_min*1.5, d_max *1.5)
    ax.set_ylim(d_min*1.5, d_max *1.5)
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title('Brain space')

    ax = fig.add_subplot(122)

    emb_Y_pred_ind = np.random.choice(np.arange(len(y_pred)), plot_settings.num_select)
    emb_Y_pred_dec_all = rotate_array(y_pred[:,[1,0]], plot_settings.angle)
    emb_Y_pred_dec = emb_Y_pred_dec_all[emb_Y_pred_ind]
    
    Y_train_ind = np.random.choice(np.arange(len(y_train)), plot_settings.num_select_train)
    emb_Y_train_all = rotate_array(y_train[:,[1,0]], plot_settings.angle)
    emb_Y_train = emb_Y_train_all[Y_train_ind]
    
    y_data_test  = [emb_Y_pred_dec[...,0], emb_Y_pred_dec[...,1]]
    
    y_data_train = [emb_Y_train[..., 0], emb_Y_train[..., 1]]
    ax.scatter(y_data_test[0], y_data_test[1], c='0.8',
               s=plot_settings.size, alpha=plot_settings.alpha,
               label="Decoding model")

    
    ax.scatter(y_data_train[0], y_data_train[1], c=[(26/255, 90/255, 46/255)], 
               s=plot_settings.size, alpha=plot_settings.alpha,
               label='Training target')
    
    if with_true_oos:
        
        #emb_Y_oos = rotate_array(y_oos[emb_Y_pred_ind][:,:2], plot_settings.angle)
        emb_Y_oos = rotate_array(y_oos[:,[1,0]], plot_settings.angle)
        x = emb_Y_oos[...,0]
        y = emb_Y_oos[...,1]
        xy = np.vstack([x,y])
        y_data_oos  =[emb_Y_oos[...,1], emb_Y_oos[...,0]]
        kde = gaussian_kde(xy)
        #y_d_min = np.min(np.concatenate([y_data_oos, y_data_test, y_data_train],axis=1))
        #y_d_max = np.max(np.concatenate([y_data_oos, y_data_test, y_data_train],axis=1))
        y_d_min = np.min(np.concatenate([emb_Y_oos, emb_Y_pred_dec_all, emb_Y_train_all]))
        y_d_max = np.max(np.concatenate([emb_Y_oos, emb_Y_pred_dec_all, emb_Y_train_all]))
        x_grid = np.linspace(y_d_min, y_d_max, 200)
        y_grid = np.linspace(y_d_min, y_d_max, 200)
        X, Y = np.meshgrid(x_grid, y_grid)

        # 各グリッドポイントでの密度を計算
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                
        #ax.scatter(y_data_oos[0], y_data_oos[1], c='b',
        #        s=plot_settings.size, alpha=0.1,
        #        label="GT random model")
        percentages = [25, 50, 75]
        levels= np.percentile(Z, percentages)
        levels = np.unique(levels)
        contour = ax.contour(X, Y, Z, levels=levels, alpha=plot_settings.alpha)
        #contour = ax.contour(X, Y, Z, levels=3, alpha=plot_settings.alpha)
        #ax.clabel(contour, inline=True, fontsize=8,)
        
        x_max_abs = np.max(np.abs(x_grid))
        y_max_abs = np.max(np.abs(y_grid))
        ax.set_xlim(x_max_abs*-1.5, x_max_abs*1.5)
        ax.set_ylim(x_max_abs*-1.5, x_max_abs*1.5)
        
    
    else:
        y_d_min = np.min(np.concatenate([y_data_test, y_data_train],axis=1))
        y_d_max = np.max(np.concatenate([y_data_test, y_data_train],axis=1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #ax.set_xlim(y_d_min*1.5, y_d_max *1.5)
    #ax.set_ylim(y_d_min*1.5, y_d_max *1.5)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("y'_1")
    ax.set_ylabel("y'_2")
    ax.set_title('Latent feature space')
    plt.tight_layout()
    plt.show()
    return fig 
    
# %%
class ODC_SimulationSettings:
    ###  Factory parameters
    ## Data structure parameter
    dim = 512 #np.max([30000, 257 * 768])
    noise_var: float = 0.25
    short_cluster_var: float = 10
    long_cluster_var: float = 100

    num_inner_clusters: int = 2
    num_outer_clusters: int = 1

    ## Linear regression model parameters
    alpha = 1.0


    ## Data parameters
    num_train: int = 50_000
    num_test = 10000#int(100 // num_inner_clusters)
    
    ## Random seed
    seed: int = 0
        
    
    
# %%


# %%

if __name__ == "__main__":
    ## main
    print('--------------------------------------------')
    save_base_dir = './results/assets/'
    assert os.path.isdir(save_base_dir)
    np.random.seed(1)
    ### training a regression model between Random data (X) to  2 cluster data  (y)
    # %%
    # Load setting 
    factory_simulator_settings = ODC_SimulationSettings


    noise_var_list = [0,0.005,  0.05, 0.1, 0.25, 0.5, 1, 10]# 0.001]
    cluster_var_ratio_list = [1/2, 1/10, 1/100]
    num_train_list = [100, 1000, 10_000]
    
    # %%
    const = factory_simulator_settings.short_cluster_var + factory_simulator_settings.long_cluster_var
    fig_list = []
    # Generate dataset via factory settings
    for noise_var, cluster_var_ratio, num_train in product(noise_var_list, cluster_var_ratio_list, num_train_list):
        cluster_var =  cluster_var_ratio/(cluster_var_ratio+1 ) * const
        inter_cluster_var= 1/(cluster_var_ratio+1 ) * const
        print(f'noise_var:{noise_var}')
        print(f'cluster_var_ratio:{cluster_var_ratio}')
        print(f'num_train:{num_train}')
        odc_factory = ClusterProjectionDataFactory(
                input_dim=factory_simulator_settings.dim,
                output_dim=factory_simulator_settings.dim,
                num_inner_clusters=factory_simulator_settings.num_inner_clusters,
                num_outer_clusters= factory_simulator_settings.num_outer_clusters,
                noise_var=noise_var, #/factory_simulator_settings.dim,
                cluster_var= cluster_var/factory_simulator_settings.dim,
                inter_cluster_var= inter_cluster_var/factory_simulator_settings.dim,
                seed=factory_simulator_settings.seed,
            )
        
       
        
        # Initialize simulator based on 
        odc_simulator = Simulator(odc_factory, 
                                factory_simulator_settings.alpha, 
                                num_train = num_train,
                                num_test = factory_simulator_settings.num_test)
        # Perform simulation: training the decoder model
        odc_simulator.train_decoding_model()
        # Perform simulation: predicting random data 
        odc_simulator.prepare_projection()

        # Plot figure 6
        plot_settings = PlotSettings()


        # Plot the results
        fig = plot_brain_and_latent_spaces(odc_simulator,plot_settings, embed=True, with_true_oos=True)
        fig_list.append(fig)
    
    # %%
    save_dir = os.path.join(save_base_dir, "fig8/sup")
    os.makedirs(save_dir, exist_ok=True)

    fig.savefig(os.path.join(save_dir, 'sup_output_dimension_collapse_2clusters.pdf'))

    print('Done!')
    

    # %%
    ### training a regression model between Random data (X) to  3 cluster data  (y)

    # Change the number of training clusters from 2 to 3
    

    odc_factory = ClusterProjectionDataFactory(
            input_dim=factory_simulator_settings.dim,
            output_dim=factory_simulator_settings.dim,
            num_inner_clusters=3,
            num_outer_clusters= factory_simulator_settings.num_outer_clusters,
            noise_var=factory_simulator_settings.noise_var,
            cluster_var= factory_simulator_settings.short_cluster_var/factory_simulator_settings.dim,
            inter_cluster_var= factory_simulator_settings.long_cluster_var/factory_simulator_settings.dim,
            seed=factory_simulator_settings.seed,
        )
    
    odc_factory = ClusterProjectionDataFactory(
            input_dim=factory_simulator_settings.dim,
            output_dim=factory_simulator_settings.dim,
            num_inner_clusters=3,
            num_outer_clusters= factory_simulator_settings.num_outer_clusters,
            noise_var=factory_simulator_settings.noise_var,
            cluster_var= factory_simulator_settings.short_cluster_var/factory_simulator_settings.dim,
            inter_cluster_var= factory_simulator_settings.long_cluster_var/factory_simulator_settings.dim,
            seed=factory_simulator_settings.seed,
        )


    odc_simulator = Simulator(odc_factory,factory_simulator_settings.alpha, 
                            num_train =  int(factory_simulator_settings.num_train), 
                            num_test = int(factory_simulator_settings.num_test))
    odc_simulator.train_decoding_model()
    odc_simulator.prepare_projection()



    # %%

    fig = plot_brain_and_latent_spaces(odc_simulator,plot_settings)
    # %%
    fig.savefig(os.path.join(save_dir, 'sup_output_dimension_collapse_3clusters.pdf'))

    print('Done!')
    
    
    # %%
    