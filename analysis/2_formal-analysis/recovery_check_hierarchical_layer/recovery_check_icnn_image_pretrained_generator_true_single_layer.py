from __future__ import annotations

from pathlib import Path
from itertools import chain
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bdpy.dl.torch.models import VGG19, layer_map, model_factory
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.dl.torch.domain import Domain, image_domain, ComposedDomain
from bdpy.recon.torch.modules import build_encoder, build_generator, TargetNormalizedMSE
from bdpy.recon.torch.modules.latent import ArbitraryLatent
from critic_image import TVloss
from bdpy.recon.torch.modules.generator import  BaseGenerator, FrozenGenerator
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

class BoundedLatent(ArbitraryLatent):
    def __init__(self, shape: tuple[int, ...], init_fn: Callable[[torch.Tensor], None],
                 upperbound: torch.Tensor, lowerbound: torch.Tensor) -> None:
    
        super().__init__(shape, init_fn)
        self.upperbound = upperbound
        self.lowerbound = lowerbound

    def generate(self) -> torch.Tensor:
        # clip upper bound
        if self.upperbound is not None:
            for latent in self._latent:
                latent.data = torch.clamp(latent, max=self.upperbound)
        # clip lower bound       
        if self.lowerbound is not None:
            for latent in self._latent:
                latent.data = torch.clamp(latent, min=self.lowerbound)
        
        return latent


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
    assets_root_path = Path(f"{cfg.output.path}")

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
        
        generator_network = model_factory(cfg.generator.name)
        generator_network.load_state_dict(torch.load(cfg.generator.parameters_file))
        generator_network.to(device)
        #generator = build_generator(...)
        generator = build_generator(generator_network, image_domain.BdPyVGGDomain(device=device, dtype=dtype) )
        #latent = ArbitraryLatent((1, 3, 224, 224), partial(nn.init.normal_, mean=0, std=1)) 
        latent_upperbound = np.loadtxt(cfg.generator.latent_upper_bound_file, delimiter=" ")
        latent_feat_num = cfg.generator.latent_shape
        latent_upperbound = torch.tensor(latent_upperbound).to(device, dtype=dtype)
        latent = BoundedLatent((batch_size, latent_feat_num[0]), partial(torch.nn.init.normal_, mean=0.0, std=0.01), 
                        latent_upperbound, 0)
        #latent = ArbitraryLatent((batch_size, latent_feat_num[0]), partial(nn.init.normal_, mean=0, std=0.01)) 
        latent.to(device)
        latent.reset_states()
        
        critic = TargetNormalizedMSE()
        critic_image = TVloss(alpha=alpha)
        #optimizer = optim.AdamW([{'params': param} for param in latent.parameters()], lr=0.01)
        optimizer = optim.AdamW(latent.parameters(), lr=1.0)
        scheduler = None
        #pipeline = FeatureInversionTask(
        #    encoder=encoder,
        #    generator=generator,
        #    latent = latent,
        #    critic=critic,
        #    optimizer=optimizer,
        #    scheduler=scheduler,
        #    num_iterations=800,

        #)
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
       