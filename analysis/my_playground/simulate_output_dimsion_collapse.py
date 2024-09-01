# %%
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.special
import scipy.stats
import torch
from torch.utils.data import Dataset, TensorDataset

from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_chunked

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
    def generate_inner(self, num: int):# -> tuple[np.ndarray, np.ndarray]:
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
    def generate_outer(self, num: int):# -> tuple[np.ndarray, np.ndarray]:
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
        raise NotImplementedError
# %%
class RondomProjectionDataFactory(DataFactory):
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
            scale = self.inter_cluster_scale
        )
        
        outer_cluster_centers = self._rng.normal(
            size=(self.num_outer_clusters, self.output_dim),
            scale = self.inter_cluster_scale)
        
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
            scale=self.cluster_scale, size=(num, self.output_dim)
        )
        # generate from random data
        x =  self._rng.normal(
            scale=self.noise_scale, size=(num, self.input_dim)
        )
        self.inner_assignments = assignments
        return x, y, assignments

    def generate_outer(self, num: int): # -> tuple[np.ndarray, np.ndarray]:
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
        x = self._rng.normal(
            scale=self.noise_scale, size=(num, self.input_dim)
        )

        return x, y,assignments
        
def generate_dataset(
    factory: DataFactory, num_train: int, num_test: int
):# -> tuple[Dataset, Dataset, Dataset]:
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
    x_train, y_train, assignments_train = factory.generate_inner(num_train)
    x_test, y_test, assignments_test = factory.generate_inner(num_test)
    x_oos, y_oos, assignments_oos = factory.generate_outer(num_test)
    
        
    train_dataset = TensorDataset(
        torch.Tensor(x_train.astype(np.float32)),
        torch.Tensor(y_train.astype(np.float32)),
        torch.Tensor(assignments_train.astype(np.int64))
    )
    test_dataset = TensorDataset(
        torch.Tensor(x_test.astype(np.float32)),
        torch.Tensor(y_test.astype(np.float32)),
        torch.Tensor(assignments_test.astype(np.int64))
    )
    oos_dataset = TensorDataset(
        torch.Tensor(x_oos.astype(np.float32)),
        torch.Tensor(y_oos.astype(np.float32)),
        torch.Tensor(assignments_oos.astype(np.int64))
    )

    return train_dataset, test_dataset, oos_dataset


def get_model(alpha: float) -> Pipeline:
    tt = TransformedTargetRegressor(
        regressor=Ridge(alpha=alpha),
       # transformer=StandardScaler()
    )
    model = Pipeline([
        #("scaler", StandardScaler()),
        ("regressor", tt)
    ])
    return model

# %%
## Factory parameters

noise_scale: float = 0.1#1.0#1.0#1.0
inter_cluster_scale: float = 1
cluster_scale: float = 0.1
    
seed = 0
## Data parameters
num_train = 10_000


## Model parameters
alpha = 1.0
# %%
###### important parameters
dim = 800
dim = 2


num_inner_clusters = 3#256
num_outer_clusters =1

num_test = 10_000//num_inner_clusters
# %%
pc_factory = RondomProjectionDataFactory(
        input_dim=dim,
        output_dim=dim,
        num_inner_clusters=num_inner_clusters,
        num_outer_clusters= num_outer_clusters,
        noise_scale=noise_scale,
        cluster_scale=np.sqrt(cluster_scale),
        inter_cluster_scale= np.sqrt(inter_cluster_scale),
        seed=seed,
    )

# %%
pc_factory.inner_cluster_centers = np.array([[0,0],
                                                                [10, 0],
                                                                [0, 10]])

pc_factory.outer_cluster_centers = np.array([[10,10],
                                                                ])
# %%
d = Simulator(pc_factory, alpha, num_train =  int(num_train), num_test = int(num_test))
# %%
d.train_decoding_model()


# %%
d.prepare_projection()
# %%
