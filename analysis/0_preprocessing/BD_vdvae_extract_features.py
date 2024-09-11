## Code is based on https://github.com/ozcelikfu/brain-diffuser/blob/main/scripts/vdvae_extract_features.py

# %%
import sys
# %%
sys.path.append('brain-diffuser/vdvae')
# %%
import torch
import numpy as np
#from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
#from apex.optimizers import FusedAdam as AdamW
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params
from image_utils import *
from model_utils import load_vaes, set_up_data
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import yaml
import pickle
from scipy.io import savemat
# %%

class batch_generator_external_image_files(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = sorted(os.listdir(self.data_path))


    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.data_path,self.im[idx])).convert('RGB')
        img = T.functional.resize(img,(64,64))
        img = torch.tensor(np.array(img)).float()
        img = img/255
        #img = img*2 - 1
        return img, self.im[idx]

    def __len__(self):
        return  len(self.im)

def load_config(config_path):
    '''Load configuration file.'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_vdvae_features(config):
    batch_size=30
    # %%
    print('Libs imported')

    H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './brain-diffuser', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': './brain-diffuser/vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    H = dotdict(H)

    H, preprocess_fn = set_up_data(H)

    print('Models is Loading')
    ema_vae = load_vaes(H)
    # %%

    data_path = config['image path']
    # %%
    #data_path = '/home/nu/data/contents_shared/NSD-stimuli/source'
    image_data = batch_generator_external_image_files(data_path)
    data_loader = DataLoader(image_data,batch_size,shuffle=False)
    # %%
    num_latents = 31
    # %%
    output_dir = os.path.join(config["output base dir"], "pytorch", 'brain_diffuser_vdvae_decoder')
    # %%
    for i,(imgs_torch_tensor, image_files) in enumerate(data_loader):
        
        # %%
        data_input, target = preprocess_fn(imgs_torch_tensor)
        # %%
        with torch.no_grad():
            print(i*batch_size)
            activations = ema_vae.encoder.forward(data_input)
            px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
        
        for i in range(num_latents):
            layer = f'dec_blocks_{i:03d}'
            
            for j, image_file in enumerate(image_files):
                f = stats[i]['z'].cpu().numpy().reshape(len(data_input),-1)[j][np.newaxis]
                
                output_file = os.path.join(
                    output_dir,
                    layer,
                    os.path.splitext(os.path.basename(image_file))[0] + '.mat'
                )
                
                if os.path.exists(output_file):
                    continue
                
                os.makedirs(os.path.join(output_dir, layer), exist_ok=True)
            
                savemat(output_file, {'feat': f})
            
        
# %%

# %%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Feature extraction')
    parser.add_argument('config', help='Configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    extract_vdvae_features(config)