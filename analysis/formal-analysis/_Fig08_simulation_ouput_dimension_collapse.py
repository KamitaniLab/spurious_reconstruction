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
from simulation_utils import DataFactory, Simulator, rotate_array


# %%
# Data class
class RandomProjectionDataFactory(DataFactory):
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
        super().__init__(input_dim, output_dim, seed=seed)

        self.num_inner_clusters = num_inner_clusters
        self.num_outer_clusters = num_outer_clusters
        self.noise_var = noise_var
        self.cluster_var = cluster_var
        self.inter_cluster_var = inter_cluster_var
        (
            self.inner_cluster_centers,
            self.outer_cluster_centers,
        ) = self._build_true_model()

    def _build_true_model(self): #-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the true model.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The true output model. The first element is the inner cluster centers, 
            and the second element is the outer cluster centers.
        """
        
        inner_cluster_centers = self._rng.normal(
            size=(self.num_inner_clusters, self.output_dim),
            scale = np.sqrt(self.inter_cluster_var)
        )
        
        outer_cluster_centers = self._rng.normal(
            size=(self.num_outer_clusters, self.output_dim),
            scale = np.sqrt(self.inter_cluster_var))
        
        return inner_cluster_centers, outer_cluster_centers

    def generate_inner(self, num: int): #-> tuple[np.ndarray, np.ndarray]:
        """
        Generate inner data.

        Parameters
        ----------
        num : int
            The number of data to generate

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The generated data. The first element is the input data and the
            second element is the output data.
        """
        assignments = self._rng.choice(self.num_inner_clusters, size=num, replace=True)
        y = self.inner_cluster_centers[assignments] + self._rng.normal(
            scale=np.sqrt(self.cluster_var), size=(num, self.output_dim)
        )
        # generate from random data
        x =  self._rng.normal(
            scale=np.sqrt(self.noise_var), size=(num, self.input_dim)
        )
        self.inner_assignments = assignments
        return x, y, assignments

    def generate_outer(self, num: int, scale_value: int = 2.3): # -> tuple[np.ndarray, np.ndarray]:
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
        y = self.outer_cluster_centers[assignments] + self._rng.normal(
            scale=np.sqrt(self.cluster_var), size=(num, self.output_dim)
        )
        x = self._rng.normal(
            scale=scale_value, size=(num, self.input_dim)
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
        self.num_select = 1000
        self.num_select_train = 20
        self.alpha = 1.0
        self.angle = -32
        self.y_axis = 0
        self.x_axis = 1
        np.random.seed(1)
    
def plot_brain_and_latent_spaces(simulator, plot_settings, embed=True):
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
        y_pred = simulator.emb_Y_oos_dec
    else:
        y_train = simulator.Y_train
        y_test = simulator.Y_test
        y_pred = simulator.y_oos_pred_dec

    # 図の作成
    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(121)
    ax.scatter(data_selection.X_test[...,plot_settings.x_axis], data_selection.X_test[...,plot_settings.y_axis], 
               #c='0.8', 
               #c=[(26/255, 90/255, 46/255)],
               c = [(230/255, 88/255, 13/255)],
               label="Random input", 
               s=plot_settings.size, alpha=plot_settings.alpha, linewidths=0.0, edgecolors='white')

    ax.scatter(data_selection.X_train[...,plot_settings.x_axis], data_selection.X_train[...,plot_settings.y_axis], 
               #c=[(26/255, 90/255, 46/255)],
               c='0.8',  
               label="Training input", s=plot_settings.size, 
               alpha=plot_settings.alpha, linewidths=0.0, edgecolors='white')

    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title('Brain space')

    ax = fig.add_subplot(122)

    emb_Y_pred_ind = np.random.choice(np.arange(len(y_pred)), plot_settings.num_select)
    emb_Y_pred_dec = rotate_array(y_pred[emb_Y_pred_ind][:,:2], plot_settings.angle)
    ax.scatter(emb_Y_pred_dec[...,plot_settings.x_axis], emb_Y_pred_dec[...,plot_settings.y_axis], 
               #c='0.8',
               c = [(230/255, 88/255, 13/255)]
               s=plot_settings.size, alpha=plot_settings.alpha,
               label="Decoding model")

    Y_train_ind = np.random.choice(np.arange(len(y_train)), plot_settings.num_select_train)
    emb_Y_train = rotate_array(y_train[Y_train_ind][:,:2], plot_settings.angle)
    ax.scatter(emb_Y_train[..., plot_settings.x_axis], emb_Y_train[..., plot_settings.y_axis], 
               #c=[(26/255, 90/255, 46/255)], c
               c = '0.8',
               s=plot_settings.size, alpha=plot_settings.alpha,
               label='Training target')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("y'_1")
    ax.set_ylabel("y'_2")
    ax.set_title('Latent feature space')
    plt.show()
    return fig 
    
# %%
class ODC_SimulationSettings:
    ###  Factory parameters
    ## Data structure parameter
    dim = 512
    noise_var: float = 0.5 
    short_cluster_var: float = 0.01
    long_cluster_var: float = 1.0

    num_inner_clusters: int = 2
    num_outer_clusters: int = 1
    
    inner_intra_ratio = 10

    ## Linear regression model parameters
    alpha = 1.0


    ## Data parameters
    num_train: int = 10_000
    num_test = int(10_000 // num_inner_clusters)
    
    ## Random seed
    seed: int = 1
    
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


    # %%
    # Generate dataset via factory settings
    odc_factory = RandomProjectionDataFactory(
            input_dim=factory_simulator_settings.dim,
            output_dim=factory_simulator_settings.dim,
            num_inner_clusters=factory_simulator_settings.num_inner_clusters,
            num_outer_clusters= factory_simulator_settings.num_outer_clusters,
            noise_var=factory_simulator_settings.noise_var,
            cluster_var= factory_simulator_settings.short_cluster_var,
            inter_cluster_var= factory_simulator_settings.long_cluster_var,
            seed=factory_simulator_settings.seed,
        )

    # %%
    # Initialize simulator based on 
    odc_simulator = Simulator(odc_factory, 
                            factory_simulator_settings.alpha, 
                            num_train = factory_simulator_settings.num_train, 
                            num_test = factory_simulator_settings.num_test)
    # Perform simulation: training the decoder model
    odc_simulator.train_decoding_model()
    # Perform simulation: predicting random data 
    odc_simulator.prepare_projection()

    # %%
    # Plot figure 6
    plot_settings = PlotSettings()



    # %%
    # Plot the results
    fig = plot_brain_and_latent_spaces(odc_simulator,plot_settings)
    
    save_dir = os.path.join(save_base_dir, "fig8")
    os.makedirs(save_dir, exist_ok=True)

    fig.savefig(os.path.join(save_dir, 'output_dimension_collapse_2clusters.pdf'))

    print('Done!')
    

    # %%
    ### training a regression model between Random data (X) to  3 cluster data  (y)

    # Change the number of training clusters from 2 to 3
    factory_simulator_settings.num_inner_clusters: int = 3

    odc_factory = RandomProjectionDataFactory(
            input_dim=factory_simulator_settings.dim,
            output_dim=factory_simulator_settings.dim,
            num_inner_clusters=factory_simulator_settings.num_inner_clusters,
            num_outer_clusters= factory_simulator_settings.num_outer_clusters,
            noise_var=factory_simulator_settings.noise_var,
            cluster_var= factory_simulator_settings.short_cluster_var,
            inter_cluster_var= factory_simulator_settings.long_cluster_var,
            seed=factory_simulator_settings.seed,
        )


    odc_simulator = Simulator(odc_factory,factory_simulator_settings.alpha, 
                            num_train =  int(factory_simulator_settings.num_train), 
                            num_test = int(factory_simulator_settings.num_test))
    odc_simulator.train_decoding_model()
    odc_simulator.prepare_projection()



    # %%

    fig = plot_brain_and_latent_spaces(odc_simulator,plot_settings)
    
    fig.savefig(os.path.join(save_dir, 'output_dimension_collapse_3clusters.pdf'))

    print('Done!')

