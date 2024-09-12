# %%
from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterator
import os 

import hdf5storage
import numpy as np
import scipy.io as sio
from itertools import product
from PIL import Image
from hydra.utils import to_absolute_path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from itertools import chain
from functools import partial

from bdpy.recon.utils import normalize_image, clip_extreme
from bdpy.dl.torch.models import VGG19, layer_map, model_factory
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.dl.torch.dataset import ImageDataset, FeaturesDataset, DecodedFeaturesDataset, RenameFeatureKeys
from bdpy.dl.torch.domain import Domain, image_domain, ComposedDomain

from bdpy.recon.torch.task import FeatureInversionTask
from bdpy.recon.torch.modules import build_encoder, build_generator, TargetNormalizedMSE
from bdpy.recon.torch.modules.latent import ArbitraryLatent
from bdpy.recon.torch.modules.generator import BaseGenerator, FrozenGenerator

import sys
sys.path.append('./analysis/1_case_study/image-reconstruction/iCNN')
from dist_loss import DistsLoss, CombinationLoss, LayerWiseDistsLoss

# %%
def load_target_statistics(
    center_path_template: str, scale_path: str, layer_path_names: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    target_center = {}
    for layer_path_name in layer_path_names:
        center_path = center_path_template.format(layer_path_name=layer_path_name)
        target_center[layer_path_name] = hdf5storage.loadmat(center_path)["y_mean"]
    target_scale = sio.loadmat(scale_path)
    return target_center, target_scale

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

# %%

class ScaleFeatures:
    def __init__(
        self,
        target_center: dict[str, np.ndarray],
        target_scale: dict[str, float],
        ddof: int = 1,
    ) -> None:
        self.target_center = target_center
        self.target_scale = target_scale
        self.ddof = ddof

    def __call__(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        scaled_features = {}
        for layer_name, feature in features.items():
            axes_to_reduce = tuple(range(1, feature.ndim)) if feature.ndim > 1 else (0,)
            feature_scale = feature.std(
                axis=axes_to_reduce, keepdims=True, ddof=self.ddof
            )
            # NOTE: mean across channels
            feature_scale = feature_scale.mean(axis=0, keepdims=True)

            scaled_feature = (
                feature - self.target_center[layer_name]
            ) / feature_scale * self.target_scale[layer_name] + self.target_center[
                layer_name
            ]
            scaled_features[layer_name] = scaled_feature[0]
        return scaled_features
    
class SimpleStackDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self._length = min(len(d) for d in datasets)  # 最小の長さを使用して、すべてのデータセットを同じ長さに保つ

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return self._length
# %%

# Setup results directory ------------------------------------------------

def run_icnn_reconstruction(config):
    #assets_root_path = Path(f"./results/playground")
    assets_root_path = Path(cfg.output.path)
    os.makedirs(assets_root_path,exist_ok=True)


    # %%
    alpha = 4.0
    beta = 1.0
    dists_alpha = {
            'classifier[6]':     0.06661523244761258,
            'classifier[3]':     0.03871265134313895,
            'classifier[0]':     0.000629031742843134,
            'features[34]': 0.5310432634795051,
            'features[32]': 0.01975314483213067,
            'features[30]': 0.714010791024532,
            'features[28]': 0.08536182104218124,
            'features[25]': 0.030798346318926202,
            'features[23]': 0.004025735147052829,
            'features[21]': 0.0021716504774059618,
            'features[19]': 0.02880295296139471,
            'features[16]': 0.014169279688225732,
            'features[14]': 0.00019573287900056505,
            'features[12]': 0.0004887923569668929,
            'features[10]': 0.006857140440209977,
            'features[7]': 0.08084213863904581,
            'features[5]': 0.00024056214287883663,
            'features[2]': 0.003886371646003732,
            'features[0]': 0.009952859626673973,
        }
    dists_beta = {
            'classifier[6]': 0.008304840607742099,
            'classifier[3]': 0.044481711593671994, 
            'classifier[0]': 0.038457933646483915, 
            'features[34]': 0.0012780195483159135, 
            'features[32]': 0.0018775814111698145,
            'features[30]': 0.5074163077203029, 
            'features[28]': 0.002337825161420017, 
            'features[25]': 0.7100372437615771,
            'features[23]': 0.5166895849277143,
            'features[21]': 0.03998274022264576,
            'features[19]': 0.04328555659354602,
            'features[16]': 0.024733951474856346,
            'features[14]': 0.0004859871528150426,
            'features[12]': 0.039778524165843814,
            'features[10]': 0.0002639605292406699,
            'features[7]': 0.02472305546171304,
            'features[5]': 0.12888847991806807,
            'features[2]': 0.008627502425502372,
            'features[0]': 0.000865427897168344
        }
    batch_size = 1
    # %%

    encoder_name = cfg.encoder.name
    device = torch.device('cuda:0')
    dtype= torch.float32

    feature_network = VGG19()

    feature_network.load_state_dict(torch.load(cfg.encoder.parameters_file))
    feature_network.to(device)
    # %%
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

    # %%
    encoder = build_encoder(feature_network, layer_names, 
                                    domain= ComposedDomain([image_domain.BdPyVGGDomain(device=device,dtype=dtype), 
                                                        image_domain.FixedResolutionDomain((224, 224))]),

                                    )
    # %%
    #generator = build_generator(...)
    
    generator_network = model_factory(cfg.generator.name)
    generator_network.load_state_dict(torch.load(cfg.generator.parameters_file))
    generator_network.to(device)


    # %%

    latent_upperbound = np.loadtxt(cfg.generator.latent_upper_bound_file, delimiter=" ")
    latent_feat_num = cfg.generator.latent_shape
    latent_upperbound = torch.tensor(latent_upperbound).to(device, dtype=dtype)
    # %%
    generator = build_generator(generator_network, image_domain.BdPyVGGDomain(device=device, dtype=dtype) )
    # %%

    latent = BoundedLatent((batch_size, latent_feat_num[0]), partial(torch.nn.init.normal_, mean=0.0, std=0.01), 
                        latent_upperbound, 0)
    latent.to(device)
    latent.reset_states()
    # %%
    critic = CombinationLoss([LayerWiseDistsLoss(alpha_dict=dists_alpha, beta_dict=dists_beta), TargetNormalizedMSE()], [alpha, beta])
        # critic_image = TVloss(alpha=1e-07)

    optimizer = optim.AdamW(latent.parameters(), lr=0.1)
    scheduler = None
    # %%
    pipeline = FeatureInversionTask(
            encoder=encoder,
            generator=generator,
            latent = latent,
            critic=critic,
            # critic_image=critic_image,
            optimizer=optimizer,
            scheduler=scheduler,
            num_iterations=800,

        )
    # %%

    # %%
    data_root_path = Path("./data") / cfg.decoded_features.name
    feature_root_path = (
        Path("/home/nu/data/contents_shared")/ cfg.decoded_features.name / "derivatives" / "features" / cfg.decoded_features.model_name
    )
    decoded_feature_root_path = (
        Path("/home/nu/data/contents_shared") / cfg.decoded_features.name
        / "derivatives"
        / "decoded_features"
        / cfg.decoded_features.decoder_setting
        / "decoded_features"
        / cfg.decoded_features.model_name
    )

    # %%
    # Load target statistics
    print("Load target statistics")
    target_path_template = (
        f"/home/nu/data/contents_shared/{cfg.decoded_features.training_dataset}/derivatives"
        f"/feature_decoders/{cfg.decoded_features.decoder_setting}/{cfg.decoded_features.model_name}"
        "/{layer_path_name}"
        f"/{cfg.decoded_features.subjects[0]}/{cfg.decoded_features.rois[0]}"
        "/model/y_mean.mat"
    )
    scale_path = (
        "./data/models_shared/caffe/VGG_ILSVRC_19_layers"
        "/estimated_feat_std"
        "/estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat"
    )
    target_center, target_scale = load_target_statistics(
        target_path_template, scale_path, layer_path_names=list(to_layer_name.keys())
    )

    # %%
    # Load features
    print("Load features")
    def compose(*transforms):
        def _compose(x):
            for transform in transforms:
                x = transform(x)
            return x

        return _compose

    images_dataset = ImageDataset(
            root_path=data_root_path / "source",
            extension=cfg.evaluation.true_image.ext,
        stimulus_names=cfg.target_images,

        )

    for subject, roi in product(cfg.decoded_features.subjects, cfg.decoded_features.rois):
        decoded_features_dataset = DecodedFeaturesDataset(
            root_path=decoded_feature_root_path,
            layer_path_names=list(to_layer_name.keys()),
            subject_id=subject,
            roi=roi,
            stimulus_names=images_dataset._stimulus_names,
            transform=compose(
                ScaleFeatures(target_center, target_scale),
                RenameFeatureKeys(to_layer_name),
            ),
        )
        # %%
        stacked_dataset = SimpleStackDataset(
                images_dataset, decoded_features_dataset
            )

        data_loader = DataLoader(
                stacked_dataset,
                batch_size=batch_size,
                num_workers=1,
            )
        # %%
        for idx, ((true_image, stimulus_names), decoded_features) in enumerate(
                data_loader
            ):
            print(stimulus_names)
            pipeline.reset_states()
            target_features = {
                k: v.to(device=device, dtype=dtype) for k, v in decoded_features.items()
            }
            generated_images = image_domain.PILDomainWithExplicitCrop().receive(pipeline(target_features))
            for i, stimulus_name in enumerate(stimulus_names):
                image = generated_images[i].detach().cpu().numpy()
                pp_image = normalize_image(clip_extreme(image, pct=4))
                # crop the image to 224x224
                h, w, _ = image.shape
                h_start = (h - 224) // 2
                w_start = (w - 224) // 2
                pp_image = pp_image[h_start : h_start + 224, w_start : w_start + 224]

                image = Image.fromarray(pp_image)
                savedir = assets_root_path / "decoded_features"
                savedir.mkdir(parents=True, exist_ok=True)
                image.save(savedir / f"{stimulus_name}.tiff")
    
# %%

# Entry point ################################################################

if __name__ == "__main__":

    cfg = init_hydra_cfg()
    run_icnn_reconstruction(cfg)
    
    