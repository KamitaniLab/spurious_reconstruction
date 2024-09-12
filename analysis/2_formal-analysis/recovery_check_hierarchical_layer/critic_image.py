from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

#import wandb

class Critic_image(nn.Module, ABC):
    """Critic for image domain

    Parameters
    ----------
    target_image : torch.Tensor
        Target image tensor.
    """

    def __init__(self, alpha: float = 1, with_wandb: bool = False) -> None:
        super().__init__()
        self.with_wandb = with_wandb
        self.alpha = alpha

    def enable_wandb(self) -> None:
        self.with_wandb = True

    @abstractmethod
    def forward(self, generated_images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic.

        Parameters
        ----------
        generated_images : torch.Tensor (batch_size, channels, height, width)
            Generated images.

        Returns
        -------
        torch.Tensor (batch_size,)
            Loss tensor.
        """
        pass
  
class TVloss(Critic_image):
    def forward(self, generated_images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic.

        Parameters
        ----------
        generated_images : torch.Tensor (batch_size, channels, height, width)
            Generated images.

        Returns
        -------
        torch.Tensor (batch_size,)
            Loss tensor.
        """
        loss = torch.sum(torch.abs(generated_images[:, :, :, :-1] - generated_images[:, :, :, 1:])) + \
               torch.sum(torch.abs(generated_images[:, :, :-1, :] - generated_images[:, :, 1:, :]))
        loss = self.alpha * loss
        if self.with_wandb:
            wandb.log({"tv_loss": loss.mean().item()})
        return loss