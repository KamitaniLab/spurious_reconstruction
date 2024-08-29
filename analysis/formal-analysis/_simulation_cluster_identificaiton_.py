# %%
import os
import pickle
from typing import Optional
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from simulation_utils import DataFactory, Simulator, get_model, compute_mse
# %%
class SimpleClusterDataFactory(DataFactory):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_inner_clusters: int,
        num_outer_clusters: int,
        noise_var: float = 0.1,
        cluster_var: float = 1.0,
        inter_cluster_var: float = 1.0,
        seed: Optional[int] = None,
    ): #-> None:
        """
        Initialize the data factory.

        SimpleClusterDataFactory generates data from a simple cluster model.
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
            The variance of the observational noise, by default 1.0
        cluster_variance : float, optional
            The variance of the cluster (cluster radius), by default 0.1
        inter_cluster_variance: float, optional
            The variance of the inter-cluster distance, by default 1.0
        seed : Optional[int], optional
            The seed for the random number generator, by default None
        """
        super().__init__(input_dim, output_dim, seed=seed)

        self.num_inner_clusters = num_inner_clusters
        self.num_outer_clusters = num_outer_clusters
        self.num_repetition = num_outer_clusters
        self.noise_var = noise_var
        self.cluster_var = cluster_var
        self.inter_cluster_var = inter_cluster_var
        

        (
            self.W,
            self.inner_cluster_centers,
            self.outer_cluster_centers,
        ) = self._build_true_model()

    def _build_true_model(self): #-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the true model.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The true model. The first element is the input matrix W, the second
            element is the inner cluster centers, and the third element is the
            outer cluster centers.
        """
        W = self._rng.normal(
            scale=1.0 / np.sqrt(self.output_dim),
            size=(self.output_dim, self.input_dim),
            
        )

        inner_cluster_centers = self._rng.normal(
            size=(self.num_inner_clusters, self.output_dim),
            scale = np.sqrt(self.inter_cluster_var)
        )

        outer_cluster_centers = self._rng.normal(
            size=(self.num_outer_clusters, self.output_dim),
            scale = np.sqrt(self.inter_cluster_var)
        )

        return W, inner_cluster_centers, outer_cluster_centers

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
        #assignments = self._rng.choice(self.num_inner_clusters, size=num, replace=True)
        # equal number of data points in each cluster
        base_assignments = np.repeat(np.arange(self.num_inner_clusters), num // self.num_inner_clusters)
        remaining_assignments = np.random.choice(self.num_inner_clusters, num % self.num_inner_clusters, replace=True)
        assignments = np.concatenate((base_assignments, remaining_assignments))
        np.random.shuffle(assignments)
        
        
        y = self.inner_cluster_centers[assignments] + self._rng.normal(
            scale=np.sqrt(self.cluster_var), size=(num, self.output_dim)
        )
        x = y @ self.W + self._rng.normal(
            scale=np.sqrt(self.noise_var), size=(num, self.input_dim)
        )

        return x, y, assignments

    def generate_outer(self, num: int): #-> tuple[np.ndarray, np.ndarray]:
        """
        Generate outer data.

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
        # equal number of data points in each cluster
        base_assignments = np.repeat(np.arange(self.num_repetition), num // self.num_repetition)
        remaining_assignments = np.random.choice(self.num_repetition, num % self.num_repetition, replace=True)
        assignments = np.concatenate((base_assignments, remaining_assignments))
        np.random.shuffle(assignments)
        
        
        y = self.outer_cluster_centers[assignments] + self._rng.normal(
            scale=np.sqrt(self.cluster_var), size=(num, self.output_dim)
        )
        x = y @ self.W + self._rng.normal(
            scale=np.sqrt(self.noise_var), size=(num, self.input_dim)
        )

        return x, y,assignments
    
# %% 
class SimpleClusterSimulator(Simulator):
    def __init__(self,data_factory, alpha=1.0, num_train =  10_000, num_test = 1_000, num_oos= None, num_oos_repetition=32):
        if num_oos == None:
            num_oos = num_test
        super().__init__(data_factory, alpha, num_train, num_test, num_oos)
        
        # select test sampples to reduce the memory
        in_dis_cluster_id = np.unique(self.assignment_test)
        
        if num_oos_repetition > len(in_dis_cluster_id):
            #warnings.warn(f"repetition ({num_oos_repetition}) is larger than the number of cluster in traininig. Setting M to the number of cluster {len(in_dis_cluster_id)}.")
            M = len(in_dis_cluster_id)
        else:
            M = num_oos_repetition

        self.selected_clusters_index = np.random.choice(in_dis_cluster_id, size=M, replace=False)
        self.select_test_cluster_assingment = np.where(np.isin(self.assignment_test, self.selected_clusters_index))[0]
        self.X_test = self.X_test[self.select_test_cluster_assingment]
        self.Y_test = self.Y_test[self.select_test_cluster_assingment]
        import copy
        self.assignment_test_original = copy.copy(self.assignment_test)
        self.assignment_test = self.assignment_test[self.select_test_cluster_assingment]

   
     
    def calculate_cluster_identification_performance(self):
        # Perform linear regression
        self.train_decoding_model()
        # get prediction results
        y_test_pred = self.y_test_pred_dec
        y_oos_pred = self.y_oos_pred_dec
        
        # get cluster assignment
        test_assignment_original = self.assignment_test_original
        test_assignment = self.assignment_test
        oos_assignment = self.assignment_oos
        
        innner_centers = self.factory.inner_cluster_centers
        outer_centers = self.factory.outer_cluster_centers
        
        test_classification_acc_list  = []
        oos_classification_acc_list = []
        # Evaluate the (expected) performance of the prediction results under two conditions: cases 
        # where the clusters to which the test data belong are similar to the training data (in-distribution) 
        # and cases with unknown clusters (out-of-distribution [ood]).
        min_iteration = np.min([self.factory.num_repetition, self.factory.num_inner_clusters])
        #print(min_iteration)
        #for ii in range(self.factory.num_repetition):
        for ii in range(min_iteration):
            positive_cluster_id = ii
            y_positive_pred = y_oos_pred[oos_assignment == positive_cluster_id]
            oos_class = np.array([int(np.max(test_assignment_original) + 1)]* y_positive_pred.shape[0])
            outer_center = outer_centers[positive_cluster_id]
            #cand center
            cand_centers = np.insert(innner_centers, innner_centers.shape[0],outer_center, axis=0)

            #test_results = compute_cluster_classification_accuracy(y_test_pred, test_assignment, cand_centers)
            # Calculate C+1-wise identification accuracy where C is the number of clusters in the training data as a negative candidates
            # and where +1 is the positive candidate, indicating the cluster to which the test data belong.
            oos_results = compute_cluster_classification_accuracy(y_positive_pred, oos_class, cand_centers)

            
            # in-distribution test 
            # the cluster center to which the test data in the ood condition belongs is also used as a negative candidate.
            positive_cluster_id = self.selected_clusters_index[ii]#np.random.choice(self.factory.num_inner_clusters)
            y_positive_pred = y_test_pred[test_assignment ==positive_cluster_id]
            positive_class = np.array([positive_cluster_id]* y_positive_pred.shape[0])

            test_results = compute_cluster_classification_accuracy(y_positive_pred, positive_class, cand_centers)
            
            mc_num = np.min([test_results.shape[0], oos_results.shape[0]])
            #print(mc_num)
            oos_classification_acc_list.append(np.mean(oos_results[:mc_num]))
            test_classification_acc_list.append(np.mean(test_results[:mc_num]))

        test_classification_acc = np.median(test_classification_acc_list)
        oos_classification_acc = np.median(oos_classification_acc_list)
        chance_performance = 1./(self.factory.num_inner_clusters+ 1 )
        
        return test_classification_acc, oos_classification_acc, chance_performance
                      
    
# %%
class CIA_SimulationSettings:
    ###  Factory parameters
    ## Data structure parameter
    #dim: int = 512
    noise_var: float = 0.25
    short_cluster_var: float = 10
    long_cluster_var: float = 100

    #num_inner_clusters: int = 2
    #num_outer_clusters: int = 32

    ## Linear regression model parameters
    alpha: float = 1.0

    #inner_intra_ratio = 10

    ## Data parameters
    num_train: int =500000
    num_test:int = 100#int(num_train // num_inner_clusters)
    
    num_repetition= 32
    
    ## Random seed
    seed: int = 0
    

def compute_cluster_classification_accuracy(y_target, y_class, cand_centers, metric='euclidean'):
    y_class = y_class.astype(int)
    dissim_mat = cdist(y_target, cand_centers, metric=metric)
    classification_results = np.argmin(dissim_mat, axis=1)

    # 形状とデータ型を確認
    assert classification_results.shape == y_class.shape, "Inconsistent shape"
    assert classification_results.dtype == y_class.dtype, "Inconsistent data type"
    
    # 要素ごとの比較
    return classification_results == y_class


# %%

def main(intra_inter_ratio):
    # %%
    factory_simulator_settings = CIA_SimulationSettings

    const = factory_simulator_settings.short_cluster_var + factory_simulator_settings.long_cluster_var #+ factory_simulator_settings.noise_var 

    print(intra_inter_ratio)

    cluster_var =  intra_inter_ratio/(intra_inter_ratio+1 ) * const
    inter_cluster_var= 1/(intra_inter_ratio+1 ) * const
    
    noise_var = factory_simulator_settings.noise_var #1/(intra_inter_ratio+1 + 1) * const
    
    print(cluster_var, inter_cluster_var, noise_var)


    dim_list = [2, 4, 8, 16, 32 , 64 ,128, 256, 512, 1024, 2048, 4096, 8192]
    
    #dim_list = [2, 4, 8, 16, 32 , 64 ,128, 256, 512, 1024, 2048] #4096, 8192]
    
    #dim_list = [2048] #4096, 8192]
    
    #dim_list = [512, 1024, 2048, 4096, 8192]
    
    
    num_inner_clusters_list = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, #2^10 
                                        ])
    

    # %%
    save_dict= {}
    for dim in tqdm(dim_list):
        # %%
        save_dict[dim] = {}
        test_classification_accuracy_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)
        oos_classification_accuracy_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)
        chance_performances =np.zeros_like(num_inner_clusters_list, dtype=np.float32)
        print(f'dim: {dim}')
        # %%
        for i, num_inner_clusters in enumerate(num_inner_clusters_list):
            print(f'inner_cluster: {num_inner_clusters}')
            # %% 
            cia_factory = SimpleClusterDataFactory(
                input_dim=dim,
                output_dim=dim,
                num_inner_clusters=num_inner_clusters, 
                num_outer_clusters= factory_simulator_settings.num_repetition,
                noise_var=noise_var,#/factory_simulator_settings.dim,
                cluster_var= cluster_var/dim,
                inter_cluster_var= inter_cluster_var/dim,
                seed=factory_simulator_settings.seed,
            )
            # %% 
            #test_scale = np.min([num_inner_clusters, factory_simulator_settings.num_repetition])
            simulator = SimpleClusterSimulator(cia_factory, 
                                    factory_simulator_settings.alpha, 
                                    num_train = factory_simulator_settings.num_train, 
                                    num_test = factory_simulator_settings.num_test* num_inner_clusters,
                                    num_oos = factory_simulator_settings.num_test*factory_simulator_settings.num_repetition,
                                    num_oos_repetition = factory_simulator_settings.num_repetition)

            # %%
            in_acc, oos_acc, chance_acc = simulator.calculate_cluster_identification_performance()
            
            # %%
            test_classification_accuracy_list[i] = in_acc
            oos_classification_accuracy_list[i] = oos_acc
            print(f'in_acc: {in_acc}')
            print(f'oos_acc: {oos_acc}')
            
            chance_performances[i] = chance_acc
        save_dict[dim]['test_classification_accuracy'] = test_classification_accuracy_list
        save_dict[dim]['oos_classification_accuracy'] = oos_classification_accuracy_list
        save_dict[dim]['chance_performances'] = chance_performances
        # Perform simulation: training the decoder model)

    # save 
    save_dir = f'./results/cluster_identification_dim/'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/cluster_identifiction_intra_inter_ratio{intra_inter_ratio}.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

# %%
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        intra_inter_ratio = float(sys.argv[1])
    else:
        intra_inter_ratio = 0.1

    main(intra_inter_ratio)
