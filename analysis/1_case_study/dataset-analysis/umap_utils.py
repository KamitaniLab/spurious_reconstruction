import pandas as pd
import numpy as np

from bdpy.dataform import Features


def extract_features(data_sets, stimulus_ids, feature_name):
    """
    Extracts and reshapes features for each condition specified in data_sets.
    
    Parameters:
        data_conditions (dict): Dictionary of data conditions and their corresponding feature paths.
        stimulus_ids (dict): Dictionary of stimulus IDs corresponding to each data condition.
        feature_name (str): Name of the feature to extract.
    
    Returns:
        dict: A dictionary containing reshaped features for each condition.
    """
    features_dict = {}
    for data_cond, feat_path in data_sets.items():
        print(feat_path)
        
        if stimulus_ids[data_cond] is not None:
            stimulus_name = pd.read_csv(stimulus_ids[data_cond], header=None)[0].values
        else:
            stimulus_name = None
        
        data_features = Features(feat_path)
        feats = data_features.get(feature_name, label=stimulus_name)
        
        feats_2d = feats.reshape(feats.shape[0], -1)
        
        num_stimulus = feats_2d.shape[0]
        print(f'{data_cond} has {num_stimulus} samples')
        
        features_dict[data_cond] = feats_2d
    
    return features_dict

def concatenate_features(features_dict):
    """
    Concatenates all feature arrays stored in the dictionary into a single numpy array.

    Parameters:
        features_dict (dict): Dictionary where each key-value pair corresponds to a condition and its associated feature array.

    Returns:
        numpy.ndarray: A single concatenated array of all feature arrays.
    """
    feat_list = [val for val in features_dict.values()]
    return np.vstack(feat_list)

def norm_feat(feats):
    m = np.mean(feats, 0)
    s  = np.std(feats, 0) + 1e-12
    
    return (feats -m)/s,m, s