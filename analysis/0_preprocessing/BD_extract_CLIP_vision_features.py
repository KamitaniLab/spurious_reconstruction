## Code is based on https://github.com/ozcelikfu/brain-diffuser/blob/main/scripts/clipvision_extract_features.py

# %%
import sys
# %%
sys.path.append('brain-diffuser/versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
#from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import torchvision.transforms as T
import yaml
from easydict import EasyDict as edict
import argparse
import os.path as osp

from scipy.io import savemat
# %%
# %%
class batch_generator_external_image_files(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = sorted(os.listdir(self.data_path))


    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.data_path,self.im[idx])).convert('RGB')
        img = T.functional.resize(img,(512,512))
        img = torch.tensor(np.array(img)).float()
        #img = img/255
        img = img*2 - 1
        return img.permute(2,0, 1), self.im[idx]

    def __len__(self):
        return  len(self.im)

def load_config(config_path):
    '''Load configuration file.'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_CLIP_vision_features(config):
    batch_size=1
    # %%
    cfgm_name = 'vd_noema'
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    # %%
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)    
    # %%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.clip = net.clip.to(device)
    # %%

    data_path = config['image path']
    # %%
    #data_path = '/home/nu/data/contents_shared/NSD-stimuli/source'
    image_data = batch_generator_external_image_files(data_path)
    data_loader = DataLoader(image_data,batch_size,shuffle=False)
    # %%
    num_embed, num_features, num_image = 257, 768, image_data.__len__()
    
    clip_feat = np.zeros((num_image,num_embed,num_features))
    # %%
    output_dir = os.path.join(config["output base dir"], "pytorch", 'brain_diffuser_versatile_diffusion')
    # %%
    layer = 'vision_encoder'
    # %%
    with torch.no_grad():
        # %%
        for i,(imgs_torch_tensor, image_files) in enumerate(data_loader):
            
            # %%
            c_v_feature = net.clip_encode_vision(imgs_torch_tensor)
            
            for j, image_file in enumerate(image_files):
                f = c_v_feature[j][np.newaxis].cpu().numpy()
                
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
    extract_CLIP_vision_features(config)