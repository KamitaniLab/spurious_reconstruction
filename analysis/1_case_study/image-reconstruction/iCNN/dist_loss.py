from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from typing import Dict, Iterable
from bdpy.task.callback import CallbackHandler, BaseCallback
from bdpy.recon.torch.modules.critic import BaseCritic, LayerWiseAverageCritic

class DistsLoss(LayerWiseAverageCritic):
    def compare_layer(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str,
        eps: float = 1e-6, alpha: float = 1.0, beta: float = 1.0
    ) -> torch.Tensor:
        if feature.ndim == 2:
            # feature.shape = (batch_size, feature_dim)
            # target_feature.shape = (batch_size, feature_dim)
            feature_mean = feature.mean(dim=1, keepdim=True) # (batch_size, 1)
            target_feature_mean = target_feature.mean(dim=1, keepdim=True) # (batch_size, 1)
            feature_var = ((feature - feature_mean) ** 2).mean(dim=1, keepdim=True) # (batch_size, 1)
            target_feature_var = ((target_feature - target_feature_mean) ** 2).mean(dim=1, keepdim=True) # (batch_size, 1)
            cov = ((feature - feature_mean) * (target_feature - target_feature_mean)).mean(dim=1, keepdim=True) # (batch_size, 1)
        else:
            # feature.shape = (batch_size, channel, height, width)
            # target_feature.shape = (batch_size, channel, height, width)
            feature_mean = feature.mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            target_feature_mean = target_feature.mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            feature_var = ((feature - feature_mean) ** 2).mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            target_feature_var = ((target_feature - target_feature_mean) ** 2).mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            #cov = ((feature - feature_mean) * (target_feature - target_feature_mean)).mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            cov = (feature * target_feature).mean(dim=[2,3], keepdim=True) - feature_mean * target_feature_mean
        s1 = (2 * feature_mean * target_feature_mean + eps) / (feature_mean ** 2 + target_feature_mean ** 2 + eps) # (batch_size, ...)
        s2 = (2 * cov + eps) / (feature_var + target_feature_var + eps) # (batch_size, ...)

        return - (alpha * s1 + beta * s2).mean(dim=tuple(range(1, s1.ndim))) / (alpha + beta)

class LayerWiseDistsLoss(LayerWiseAverageCritic):
    def __init__(self, callbacks: BaseCallback | Iterable[BaseCallback] | None =None, alpha_dict: Dict = {}, beta_dict: Dict= {}) -> None:
        super(LayerWiseDistsLoss, self).__init__(callbacks)
        self.alpha_dict = alpha_dict
        self.beta_dict = beta_dict
    def compare_layer(
        self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str,eps: float = 1e-6,
         
    ) -> torch.Tensor:
        if feature.ndim == 2:
            # feature.shape = (batch_size, feature_dim)
            # target_feature.shape = (batch_size, feature_dim)
            feature_mean = feature.mean(dim=1, keepdim=True) # (batch_size, 1)
            target_feature_mean = target_feature.mean(dim=1, keepdim=True) # (batch_size, 1)
            feature_var = ((feature - feature_mean) ** 2).mean(dim=1, keepdim=True) # (batch_size, 1)
            target_feature_var = ((target_feature - target_feature_mean) ** 2).mean(dim=1, keepdim=True) # (batch_size, 1)
            cov = ((feature - feature_mean) * (target_feature - target_feature_mean)).mean(dim=1, keepdim=True) # (batch_size, 1)
        else:
            # feature.shape = (batch_size, channel, height, width)
            # target_feature.shape = (batch_size, channel, height, width)
            feature_mean = feature.mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            target_feature_mean = target_feature.mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            feature_var = ((feature - feature_mean) ** 2).mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            target_feature_var = ((target_feature - target_feature_mean) ** 2).mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            #cov = ((feature - feature_mean) * (target_feature - target_feature_mean)).mean(dim=[2,3], keepdim=True) # (batch_size, channel, 1, 1)
            cov = (feature * target_feature).mean(dim=[2,3], keepdim=True) - feature_mean * target_feature_mean
        s1 = (2 * feature_mean * target_feature_mean + eps) / (feature_mean ** 2 + target_feature_mean ** 2 + eps) # (batch_size, ...)
        s2 = (2 * cov + eps) / (feature_var + target_feature_var + eps) # (batch_size, ...)

        return - (self.alpha_dict[layer_name] * s1 + self.beta_dict[layer_name] * s2).mean(dim=tuple(range(1, s1.ndim))) / (self.alpha_dict[layer_name] + self.beta_dict[layer_name])

    
class CombinationLoss(LayerWiseAverageCritic):
    """Combination of other Critics"""

    def __init__(self, critics: list[BaseCritic], weights: list[float]) -> None:
        super().__init__()
        self.critics = critics
        self.weights = weights

    def compare_layer(
            self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
            ) -> torch.Tensor:
        loss = 0.0
        for critic, weight in zip(self.critics, self.weights):
            loss += weight * critic.compare_layer(feature, target_feature, layer_name)
        return loss

    
    