from __future__ import annotations

from pathlib import Path
from itertools import chain
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bdpy.dl.torch.models import VGG19, layer_map
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.dl.torch.domain import Domain, image_domain, ComposedDomain
from bdpy.recon.torch.modules import build_encoder, build_generator, TargetNormalizedMSE
from bdpy.recon.torch.modules.latent import ArbitraryLatent
from critic_image import TVloss
from bdpy.recon.torch.modules.generator import BareGenerator, BaseGenerator
from bdpy.dl.torch.dataset import ImageDataset, FeaturesDataset, DecodedFeaturesDataset, RenameFeatureKeys
from bdpy.recon.torch.task import FeatureInversionTask
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SimpleStackDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self._length = min(len(d) for d in datasets)  # 最小の長さを使用して、すべてのデータセットを同じ長さに保つ

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return self._length
    
class FeatureInversionTask_withCriticImage(FeatureInversionTask):
    
    def __init__(self, encoder, generator, latent, critic, optimizer, scheduler, num_iterations, critic_image, callbacks=None):
        super().__init__(encoder, generator, latent, critic, optimizer, scheduler, num_iterations, callbacks=callbacks)
        self._critic_image = critic_image
        
    def __call__(self,
        target_features) -> torch.Tensor:
    
        return self.run(target_features)
    
    def run(self, target_features) -> torch.Tensor:
        self._callback_handler.fire("on_task_start")
        self.reset_states()
        for step in range(self._num_iterations):
            self._callback_handler.fire("on_iteration_start", step=step)
            self._optimizer.zero_grad()

            latent = self._latent()
            generated_image = self._generator(latent)
            self._callback_handler.fire(
                "on_image_generated", step=step, image=generated_image.clone().detach()
            )

            features = self._encoder(generated_image)

            loss = self._critic(features, target_features)
            if self._critic_image is not None:
                loss += self._critic_image(generated_image)
            self._callback_handler.fire(
                "on_loss_calculated", step=step, loss=loss.clone().detach()
            )
            loss.backward()

            self._optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()

            self._callback_handler.fire("on_iteration_end", step=step)
            # pixel decay
            #self._latent._latent.data = self._latent._latent.data * 0.99
        generated_image = self._generator(self._latent()).detach()

        self._callback_handler.fire("on_task_end")
        return generated_image


class NoGenerator(BaseGenerator):
    def __init__(
        self,
        image_shape: tuple[int, int],
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._image_shape = image_shape
        self._batch_size = batch_size
        self._latent_list = [nn.Parameter(torch.empty([1, 3, *image_shape], **factory_kwargs)) for _ in range(batch_size)]
        self.domain = image_domain.Zero2OneImageDomain()
        self.reset_states()

    def reset_states(self) -> None:
        pass

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        image = latent#torch.cat(self._latent_list, dim=0)
        image = torch.clamp(image, 0.0, 1.0)
        return self.domain.send(image)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator."""
        return iter(self._latent_list)

def run_layerwise_recovery_check(cfg):
    
    # Configuration
    print("Configuration")
    batch_size = 1
    alpha = 1e-7
    #layer = args.layer
    # Setup
    print("Setup")

    encoder_name = cfg.encoder.name
    device = torch.device('cuda:0')
    dtype= torch.float32
    feature_network = VGG19()
    feature_network.load_state_dict(torch.load(cfg.encoder.parameters_file))
    feature_network.to(device)
    to_layer_name = layer_map(encoder_name)
    # exclude the key starting from "key"
    to_layer_name = {k: v for k, v in to_layer_name.items() if "relu" not in k}
    
    to_path_name = {
            layer_name: layer_path_name
            for layer_name, layer_path_name in zip(
                to_layer_name.values(), to_layer_name.keys()
            )
        }
    layer_names = list(to_layer_name.values())
    # Remove relu layer
    layer_names = [layer_name for layer_name in layer_names if "relu" not in to_path_name[layer_name]]

    # set up features
    data_root_path = Path("./data") / cfg.dataset.name

    feature_root_path = (
        Path("./data")/  cfg.dataset.name / "derivatives" / "features" / "caffe/VGG_ILSVRC_19_layers"
    )
    #assets_root_path = Path(f"./results/recovery_check/{cfg.dataset.name}")
    assert_root_path = assets_root_path = Path(f"{cfg.output.path}")
    # Load features
    print("Load features")
    images_dataset = ImageDataset(
        root_path=data_root_path / "source",
        stimulus_names=cfg.target_images,
        extension=cfg.true_image.ext,
    )
    
    features_dataset = FeaturesDataset(
        root_path=feature_root_path,
        layer_path_names=list(to_layer_name.keys()),
        stimulus_names=images_dataset._stimulus_names,
        transform=RenameFeatureKeys(to_layer_name),
    )
    
    stacked_dataset = SimpleStackDataset(
        images_dataset, features_dataset
    )

    data_loader = DataLoader(
        stacked_dataset,
        batch_size=batch_size,
        num_workers=1,
    )


    for layer in layer_names:
        print(layer)
        encoder = build_encoder(feature_network, [layer], 
                                        domain= ComposedDomain([image_domain.BdPyVGGDomain(device=device,dtype=dtype), 
                                                            image_domain.FixedResolutionDomain((224, 224))]),)
        generator = NoGenerator((3, 224, 224), 1, device=device, dtype=dtype)
        #latent = ArbitraryLatent((1, 3, 224, 224), partial(nn.init.normal_, mean=0, std=1)) 
        latent = ArbitraryLatent((1, 3, 224, 224), partial(torch.nn.init.constant_, val=0.5)) 
        latent.to(device)
        
        critic = TargetNormalizedMSE()
        critic_image = TVloss(alpha=alpha)
        #optimizer = optim.AdamW([{'params': param} for param in latent.parameters()], lr=0.01)
        optimizer = torch.optim.AdamW(latent.parameters(), lr=0.01)
        scheduler = None

        pipeline = FeatureInversionTask_withCriticImage(
            encoder, generator, latent, critic, optimizer,
                    num_iterations=800,#num_iteration,
                    scheduler = None,
                    critic_image=critic_image,
                    #callbacks=[CUILoggingCallback()],
                )#.register_callback(callback)
        #pipeline.register_callback(callback)

        # Run
        print("Run")
        for idx, ((true_image, stimulus_names), features) in enumerate(
            data_loader
        ):
            print(f"batch [{idx+1}/{len(data_loader)}]")

            print("Reconstructing true image")
            target_feature_type = "features"
            
            pipeline.reset_states()
            target_features = {
                k: v.to(device=device, dtype=dtype) for k, v in features.items()
            }
            
            generated_images = image_domain.PILDomainWithExplicitCrop().receive(pipeline(target_features))
            for i, stimulus_name in enumerate(stimulus_names):
                image = Image.fromarray(
                    generated_images[i].detach().cpu().numpy().astype(np.uint8)
                )
                savedir = assets_root_path / target_feature_type / to_path_name[layer]
                savedir.mkdir(parents=True, exist_ok=True)
                image.save(savedir / f"{stimulus_name}.tiff")
            
       
if __name__ == "__main__":

    cfg = init_hydra_cfg()
    run_layerwise_recovery_check(cfg)
       