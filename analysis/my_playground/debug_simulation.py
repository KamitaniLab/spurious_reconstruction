# %%
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import os
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, TensorDataset
from scipy.spatial.distance import cdist

# 
print("modify dimension effect")
# %%
class DataFactory(ABC):
    """Abstract class for data factory."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the data factory.

        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        seed : Optional[int], optional
            The seed for the random number generator, by default None
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def generate_inner(self, num: int) -> tuple[np.ndarray, np.ndarray]:
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
        raise NotImplementedError

    @abstractmethod
    def generate_outer(self, num: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate outer data.

        Parameters
        ----------
        num : int
            The number of data to generate

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The generated data. The first element is the input data and the
            second element is the output data.
            Also outputed where the 
        """
        raise NotImplementedError


class SimpleClusterDataFactory(DataFactory):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_inner_clusters: int,
        num_outer_clusters: int,
        noise_scale: float = 0.1,
        cluster_scale: float = 1.0,
        inter_cluster_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
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
        noise_scale : float, optional
            The scale of the observational noise, by default 1.0
        cluster_scale : float, optional
            The scale of the cluster (cluster radius), by default 0.1
        inter_cluster_scale": float, optional
            The scale of the inter-cluster distance, by default 1.0
        seed : Optional[int], optional
            The seed for the random number generator, by default None
        """
        super().__init__(input_dim, output_dim, seed=seed)

        self.num_inner_clusters = num_inner_clusters
        self.num_outer_clusters = num_outer_clusters
        self.noise_scale = noise_scale
        self.cluster_scale = cluster_scale
        self.inter_cluster_scale = inter_cluster_scale

        (
            self.W,
            self.inner_cluster_centers,
            self.outer_cluster_centers,
        ) = self._build_true_model()

    def _build_true_model(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            scale = self.inter_cluster_scale
        )

        outer_cluster_centers = self._rng.normal(
            size=(self.num_outer_clusters, self.output_dim),
            scale = self.inter_cluster_scale
        )

        return W, inner_cluster_centers, outer_cluster_centers

    def generate_inner(self, num: int) -> tuple[np.ndarray, np.ndarray]:
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
            scale=self.cluster_scale, size=(num, self.output_dim)
        )
        x = y @ self.W + self._rng.normal(
            scale=self.noise_scale, size=(num, self.input_dim)
        )

        return x, y, assignments

    def generate_outer(self, num: int) -> tuple[np.ndarray, np.ndarray]:
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
        assignments = self._rng.choice(self.num_outer_clusters, size=num, replace=True)
        y = self.outer_cluster_centers[assignments] + self._rng.normal(
            scale=self.cluster_scale, size=(num, self.output_dim)
        )
        x = y @ self.W + self._rng.normal(
            scale=self.noise_scale, size=(num, self.input_dim)
        )

        return x, y,assignments


def generate_dataset(
    factory: DataFactory, num_train: int, num_test: int
) -> tuple[Dataset, Dataset, Dataset]:
    """Generate pytorch datasets from the given data factory.

    Parameters
    ----------
    factory : DataFactory
        The data factory
    num_train : int
        The number of training data
    num_test : int
        The number of test (both in-distribution/out-of-subspace) data

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        The generated datasets. The first element is the training dataset, the
        second element is the in-distribution test dataset, and the third
        element is the out-of-subspace test dataset.
    """
    x_train, y_train, _  = factory.generate_inner(num_train)
    x_test, y_test, test_assignment = factory.generate_inner(num_test)
    x_oos, y_oos, oos_assignment = factory.generate_outer(num_test)

    train_dataset = TensorDataset(
        torch.Tensor(x_train.astype(np.float32)),
        torch.Tensor(y_train.astype(np.float32))
    )
    test_dataset = TensorDataset(
        torch.Tensor(x_test.astype(np.float32)),
        torch.Tensor(y_test.astype(np.float32)),
        torch.Tensor(test_assignment),
    )
    oos_dataset = TensorDataset(
        torch.Tensor(x_oos.astype(np.float32)),
        torch.Tensor(y_oos.astype(np.float32)),
        torch.Tensor(oos_assignment),
    )

    return train_dataset, test_dataset, oos_dataset


def get_model(alpha: float) -> Pipeline:
    tt = TransformedTargetRegressor(
        regressor=Ridge(alpha=alpha),
        transformer=StandardScaler()
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", tt)
    ])
    return model

def compute_mse(target: np.ndarray, pred: np.ndarray, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    if scaler is not None:
        target = scaler.transform(target)  # TODO: inverse_transform? -> OK!
        pred = scaler.transform(pred)
    return ((target - pred) ** 2).mean(axis=-1)


def visualize_embedding(emb: np.ndarray, color: str, label: str, alpha: float = 0.8, s: float = 0.1, marker: str = '.'):
    plt.scatter(emb[..., 0], emb[..., 1], marker=marker, c=color, s=s, alpha=alpha, label=label)


def compute_novel_classification_accuracy(
        candidates_index: int,
        target: np.ndarray,
        pred: np.ndarray,
        top_k: int = 1
    ) -> float:
    """Compute the C+1-wise identification accuracy.
    
    Parameters
    ----------
    candidates_index : faiss.Index
        The knn index of the candidate data having shape (num_candidates, output_dim)
    target : np.ndarray
        The target output having shape (num_test, output_dim)
    pred : np.ndarray
        The predicted output having shape (num_test, output_dim)
    top_k : int, optional
        The number of top candidates to consider, by default 1
    """

    sqdists, _ = candidates_index.search(pred, k=top_k)
    worst_sqdists_among_candidates = sqdists[..., -1]  # (num_test,)
    sqdists_to_target = ((target - pred) ** 2).sum(axis=-1)  # (num_test,)
    identification_accuracy = (sqdists_to_target < worst_sqdists_among_candidates).mean()
    return identification_accuracy

def compute_cluster_classification_accuracy(y_target, y_class, cand_centers, metric='euclidean'):
    y_class = y_class.astype(int)
    dissim_mat = cdist(y_target, cand_centers, metric=metric)
    classification_results = np.argmin(dissim_mat, axis=1)

    # 形状とデータ型を確認
    assert classification_results.shape == y_class.shape, "形状が一致しません。"
    assert classification_results.dtype == y_class.dtype, "データ型が一致しません。"
    
    # 要素ごとの比較
    return classification_results == y_class

# %%
## Factory parameters
# dim = 800

noise_scale = 0.5#0.1#2.0
cluster_scale = 0.1
inter_cluster_scale = 1.0

num_outer_clusters = 32
seed = 0

## Data parameters
num_train = 5000 * 100
num_test = 10_000

## Model parameters
alpha = 1.0

num_inner_clusters_list = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, #2^10 
                                    #2048, 4096, # 2^12,
                                    #4096, 8192, 16384, 
                                    #4096, 8192, 16384, 32768, 65536, # 2^16
                                    #131072, 262144, 524288
                                    ])


inner_intra_ratio=10.0
const = cluster_scale **2 + inter_cluster_scale ** 2

num_inner_clusters_list_ = [4, 32, 64, 128, 256, 1024]
num_inner_cluster_dict = {}

    
#  dim change while keeping inner_intra_ratio = 10

dim_list = [2, 4, 8, 16, 32 , 64 ,128, 256, 512, 1024, 2048, 4096, 8192 ]
inner_intra_ratio=10 

scale = 100
cluster_scale=0.1 * scale
inter_cluster_scale = 1.0 * scale

print(cluster_scale/ inter_cluster_scale)

test_accuracy_array = np.zeros([len(dim_list), len(num_inner_clusters_list)])

oos_accuracy_array = np.zeros([len(dim_list), len(num_inner_clusters_list)])
chance_array = np.zeros([len(dim_list), len(num_inner_clusters_list)])
# %%
save_dict= {}
for j, dim in enumerate(dim_list):
    save_dict[dim] = {}
    print(f'{dim}')
    test_mse_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)
    oos_mse_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)
    uniform_mse_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)
    test_classification_accuracy_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)
    oos_classification_accuracy_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)
    uniform_classification_accuracy_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)

    gt_uniform_classification_accuracy_list = np.zeros_like(num_inner_clusters_list, dtype=np.float32)

    #pc_eigenvalue_list = np.zeros_like([num_inner_clusters_list, dim], dtype=np.float32)
    pc_eigenvalue_list = np.zeros([len(num_inner_clusters_list), dim], dtype=np.float32)
    pc_eigenvalue_list_raw = np.zeros([len(num_inner_clusters_list), dim], dtype=np.float32)

    chance_performances =np.zeros_like(num_inner_clusters_list, dtype=np.float32)
   
    for i, num_inner_clusters in enumerate(tqdm(num_inner_clusters_list)):
        
        # %%
        #analysis_condition = f'{analysis_base_condition}-num_train_clusters-{num_inner_clusters}'
        #print(num_inner_clusters)
        factory = SimpleClusterDataFactory(
            input_dim=dim,
            output_dim=dim,
            num_inner_clusters=num_inner_clusters,
            num_outer_clusters=num_outer_clusters,
            noise_scale=noise_scale,
            cluster_scale=np.sqrt(cluster_scale/(dim)),
            inter_cluster_scale=np.sqrt(inter_cluster_scale/(dim) ),
            seed=seed,
        )
        # %% 

        train_dataset, test_dataset, oos_dataset = generate_dataset(
            factory, num_train, num_test
        )
        
        reg = get_model(alpha)
        reg.fit(train_dataset.tensors[0].numpy(), train_dataset.tensors[1].numpy())
        
    

        y_test_pred = reg.predict(test_dataset.tensors[0].numpy())
        y_oos_pred = reg.predict(oos_dataset.tensors[0].numpy())
        #y_uniform_pred = reg.predict(X_uniform)

        test_mse = compute_mse(test_dataset.tensors[1].numpy(), y_test_pred).mean()
        oos_mse = compute_mse(oos_dataset.tensors[1].numpy(), y_oos_pred).mean()
        #uniform_mse = compute_mse(y_uniform, y_uniform_pred).mean()
        test_assignment = test_dataset.tensors[2].numpy()
        oos_assignment = oos_dataset.tensors[2].numpy()


        innner_centers = factory.inner_cluster_centers
        outer_centers = factory.outer_cluster_centers

        test_classification_acc_list  = []
        oos_classification_acc_list = []
        uniform_classification_acc_list = []
        gt_uniform_classification_acc_list = []
        for ii in range(factory.num_outer_clusters):
            y_oosi_pred = y_oos_pred[oos_dataset.tensors[2].numpy() == ii]
            oos_class = np.array([int(np.max(test_assignment) + 1)]* y_oosi_pred.shape[0])
            outer_center = outer_centers[ii]
            #cand center
            cand_centers = np.insert(innner_centers, innner_centers.shape[0],outer_center, axis=0)

            test_results = compute_cluster_classification_accuracy(y_test_pred, test_assignment, cand_centers)
            oos_results = compute_cluster_classification_accuracy(y_oosi_pred, oos_class, cand_centers)
            #uniform_results = compute_cluster_classification_accuracy(y_uniform_pred, oos_class, cand_centers)
            #gt_uniform_results = compute_cluster_classification_accuracy(y_uniform, oos_class, cand_centers)
            test_classification_acc_list.append(np.mean(test_results))
            oos_classification_acc_list.append(np.mean(oos_results))
            #uniform_classification_acc_list.append(np.mean(uniform_results))
            #gt_uniform_classification_acc_list.append(np.mean(gt_uniform_results))
        test_classification_acc = np.median(test_classification_acc_list)
        oos_classification_acc = np.median(oos_classification_acc_list)
        #uniform_classification_acc = np.median(uniform_classification_acc_list)
        #gt_uniform_classification_acc = np.median(gt_uniform_classification_acc_list)

        test_mse_list[i] = test_mse
        oos_mse_list[i] = oos_mse
        #uniform_mse_list[i] = uniform_mse

        test_classification_accuracy_list[i] = test_classification_acc
        oos_classification_accuracy_list[i] = oos_classification_acc
        #uniform_classification_accuracy_list[i] = uniform_classification_acc
        #gt_uniform_classification_accuracy_list[i] = gt_uniform_classification_acc
        chance_performances[i] = 1./(num_inner_clusters+ 1 )
        
        
        
    save_dict[dim]['test_classification_accuracy'] = test_classification_accuracy_list
    save_dict[dim]['oos_classification_accuracy'] = oos_classification_accuracy_list
    save_dict[dim]['chance_performances'] = chance_performances
    
    print(test_classification_accuracy_list)
    print(oos_classification_accuracy_list)
    
    #assert 1== 0
# save save_dict
save_dir = './results/cluster_identification_dim_change_input_dim_v3'
os.makedirs(save_dir, exist_ok=True)
with open(f'{save_dir}/save_dict_10_1000000_samples_v3_small_noise.pkl', 'wb') as f:
    pickle.dump(save_dict, f)

    
     
# %%
