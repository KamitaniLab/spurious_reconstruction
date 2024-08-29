# %%
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import scipy.stats
import torch
from torch.utils.data import Dataset, TensorDataset

from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_chunked


## Common class in Figures 8 and 9.

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
    

class Simulator():
    
    def __init__(self, data_factory, alpha=1.0, num_train =  10_000, num_test = 1_000, num_oos=None):
        if num_oos == None:
            num_oos = num_test
        train_dataset, test_dataset, oos_dataset = generate_dataset(
                                data_factory, num_train, num_test, num_oos
                                                     )
        self.factory = data_factory
        self.X_train = train_dataset.tensors[0].numpy()
        self.X_test = test_dataset.tensors[0].numpy()
        self.X_oos = oos_dataset.tensors[0].numpy()
        
        self.Y_train = train_dataset.tensors[1].numpy()
        self.Y_test = test_dataset.tensors[1].numpy()
        self.Y_oos = oos_dataset.tensors[1].numpy()
        self.emb_method = None
        
        self.assignment_train = train_dataset.tensors[2].numpy()
        self.assignment_test = test_dataset.tensors[2].numpy()
        self.assignment_oos = oos_dataset.tensors[2].numpy()
        
        self.num_train = num_train
        self.num_test = num_test
        
        self.dim = self.Y_test.shape[1]
        
        self.decoder = get_model(alpha)
        
        self.x_mean =self.X_train.mean(0)
        self.x_std = self.X_train.std(0)
        self.y_mean = self.Y_train.mean(0)
        self.y_std = self.Y_train.std(0)
        
        self.Y_prior = np.cov(self.Y_train, rowvar=False)#X_train_gram / ( len(self.X_train) -1) 
        
    def train_decoding_model(self):
        self.decoder.fit(self.X_train, self.Y_train)
        self.W_dec = self.decoder.named_steps['regressor'].regressor_.coef_
        self.y_test_pred_dec = self.decoder.predict(self.X_test)
        self.y_oos_pred_dec = self.decoder.predict(self.X_oos)
        
        
    def calcaulte_performance(self, pred_test, test, pred_oos, oos):
        
         # calcuate mse 
        mse_test = np.mean(compute_mse(pred_test, test))
        mse_oos = np.mean(compute_mse(pred_oos, oos))
        
        print(f'test {mse_test},  oos {mse_oos}')
        return mse_test, mse_oos
        
    def prepare_projection(self, method='pca'):

        concat_Y= self.Y_train
    
        concat_Y, mean_y, std_y = norm_feat(concat_Y)
        self.mean_y = mean_y
        self.std_y = std_y
        self.emb_method = method
        if method == 'pca':
            self.embed = PCA()

        else: 
            raise NotImplementedError("Other embedding method is not implemented yet.")
        self.embed.fit(concat_Y)

        self.emb_Y_train = self.embed.transform( (self.Y_train - mean_y)/ std_y)
        self.emb_Y_test =self.embed.transform((self.Y_test - mean_y)/ std_y)
        self.emb_Y_oos = self.embed.transform((self.Y_oos - mean_y)/ std_y)
        self.emb_Y_test_dec = self.embed.transform( (self.y_test_pred_dec - mean_y) / std_y)
        self.emb_Y_oos_dec = self.embed.transform( (self.y_oos_pred_dec - mean_y) /std_y)

        #self.emb_Y_gauss_sanity = self.embed.transform( (self.y_gauss_sanity - mean_y) /std_y)

        print('done projection')
        
    

def generate_dataset(
    factory: DataFactory, num_train: int, num_test: int, num_oos:int =None
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
    num_oos : int, optional
        The number of out-of-sample data, by default None 
        if num_ood is not specified (e.g., None), the function will generate the same_values with num_test.
    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        The generated datasets. The first element is the training dataset, the
        second element is the in-distribution test dataset, and the third
        element is the out-of-subspace test dataset.
    """
    x_train, y_train, assignments_train = factory.generate_inner(num_train)
    x_test, y_test, assignments_test = factory.generate_inner(num_test)
    if num_oos == None:
        num_oos = num_test
    x_oos, y_oos, assignments_oos = factory.generate_outer(num_oos)
    
    
        
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

## functions 

def compute_mse(target: np.ndarray, pred: np.ndarray, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """
    Compute the Mean Squared Error (MSE) between the target and prediction arrays.

    Parameters
    ----------
    target : np.ndarray
        The true values.
    pred : np.ndarray
        The predicted values.
    scaler : Optional[StandardScaler], optional
        A scaler for normalizing the data, by default None.

    Returns
    -------
    np.ndarray
        The computed MSE for each sample.
    """
    if scaler is not None:
        target = scaler.transform(target)
        pred = scaler.transform(pred)
    return ((target - pred) ** 2).mean(axis=-1)

def get_model(alpha: float) -> Pipeline:
    """
    Get a model pipeline with a Ridge regressor and optional scaling.

    Parameters
    ----------
    alpha : float
        Regularization strength of the Ridge regressor.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with a Ridge regression model possibly including a scaler.
    """
    tt = TransformedTargetRegressor(
        regressor=Ridge(alpha=alpha),
        transformer=StandardScaler()
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", tt)
    ])
    return model

# normalization befor
def norm_feat(feats: np.array):
    """
    Normalize the features by subtracting the mean and dividing by the standard deviation.

    Parameters
    ----------
    feats : np.array
        The features to be normalized.

    Returns
    -------
    tuple
        A tuple containing the normalized features, the mean, and the standard deviation.
    """
    m = np.mean(feats, 0)
    s  = np.std(feats, 0) + 1e-12
    
    return (feats -m)/s,m, s


def rotate_array(data: np.array, degrees: float):
    """
    Rotate a numpy array of points by a specified number of degrees.

    Parameters
    ----------
    data : np.array
        A numpy array of shape (n, 2), where each row represents a point (x, y).
    degrees : float
        The angle in degrees by which the array should be rotated.

    Returns
    -------
    np.array
        A numpy array of the rotated points.
    """
    # Convert degrees to radians
    radians = np.deg2rad(degrees)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                [np.sin(radians), np.cos(radians)]])

    # Rotate the data points
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data