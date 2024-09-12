# %%
import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
import torchvision.transforms as T
from scipy.io import savemat
from glob import glob
import pandas as pd
# %%

#def extract_NSD_CLIP_text_features():
def extract_Deeprecon_CLIP_text_features():
    cfgm_name = 'vd_noema'
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)   
    # %%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.clip = net.clip.to(device)

    # %%
    num_embed, num_features = 77, 768
    layer = 'text_encoder'
    # Deeprecon test setting
    image_list, df, save_base_path = Deeprecon_test_setting()
    
    output_dir = os.path.join(save_base_path, "pytorch", 'brain_diffuser_versatile_diffusion')
    print('Extract text features from Deeprecon test images')
    extract_CLIP_text_features(image_list, df, output_dir, layer, net)
    print('Done')
    # %%
    
    # Deeprecon traininig setting
    training_image_list, df, save_base_path_training = Deeprecon_training_setting()
    output_dir_training = os.path.join(save_base_path_training, "pytorch", 'brain_diffuser_versatile_diffusion')
    print('Extract text features from Deeprecon training images')
    extract_CLIP_text_features(training_image_list, df, output_dir_training, layer, net)
    print('Done')
    
        
def Deeprecon_test_setting():
    original_DeepreconTest_dir = './data/ImageNetTest'
    image_list = glob(os.path.join(original_DeepreconTest_dir, 'source', '*.JPEG'))
    # we didn't share the test annotation file due to the possibility to traing the future AI models.
    df = pd.read_csv(os.path.join(original_DeepreconTest_dir, 'derivatives/captions/amt_20181204/amt_20181204.csv'))
    save_base_path = "./data/ImageNetTest/derivatives/features"
    return image_list, df, save_base_path
    
def Deeprecon_training_setting():
    original_DeepreconTraining_dir = './data/contents_shared/ImageNetTraining'
    image_list = glob(os.path.join(original_DeepreconTraining_dir, 'source', '*.JPEG'))
    df = pd.read_csv(os.path.join(original_DeepreconTraining_dir, 'derivatives/captions/amt_20181204/amt_20181204.csv'))
    save_base_path = "./data/ImageNetTraining/derivatives/features"
    return image_list, df, save_base_path
    
def extract_CLIP_text_features(image_list, df, output_dir, layer, net):
    for image in image_list:
        file_name = os.path.basename(image).split('.')[0]
        caption_texts = df[df['content_id'] == file_name]['caption'].values
        #print(img)
        out_text_list  =[]
        # %%
        for cap in caption_texts:
            c_t_feature = net.clip_encode_text(cap)
            out_text_list.append(c_t_feature.cpu().detach().numpy())
        out_text_list = np.array(out_text_list).squeeze() #[5, 77, 768]
        
        # %%
        output_file = os.path.join(
            output_dir,
            layer,
            file_name + '.mat'
        )
        if os.path.exists(output_file):
            continue
        
        os.makedirs(os.path.join(output_dir, layer), exist_ok=True)

        savemat(output_file, dict([('feat', np.mean(out_text_list, 0, keepdims=True)),
            ('raw_feat', out_text_list)]) )
    
            
# %%

# %%%
if __name__ == '__main__':

    extract_Deeprecon_CLIP_text_features()
