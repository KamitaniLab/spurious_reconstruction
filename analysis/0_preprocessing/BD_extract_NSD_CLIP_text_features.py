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
from nsd_access.nsd_access.nsda import NSDAccess
from scipy.io import savemat
# %%

def extract_NSD_CLIP_text_features():

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

    origina_nsd_dir = '[PATH_TO_NSD_DATASET_DIR]' #'/home/nu/data/fmri_shared/public/NSD/latest/'
    nsda = NSDAccess(origina_nsd_dir)

    output_dir = os.path.join("./data/NSD-stimuli/derivatives/features/", "pytorch", 'brain_diffuser_versatile_diffusion')
    layer = 'text_encoder'
    # %%
    total_num = 73000
    for img in range(total_num):
        file_name = f'nsd{img+1:05}'+ '.mat'
        ci = nsda.read_image_coco_info([img], info_type='captions', show_annot=True)
        #print(img)
        out_text_list  =[]
        # %%

        for annotator in ci:
            cap = annotator['caption']
            
            c_v_feature = net.clip_encode_text(cap)
            out_text_list.append(c_v_feature.cpu().detach().numpy())
        out_text_list = np.array(out_text_list).squeeze() #[5, 77, 768]
        
        # %%
        output_file = os.path.join(
            output_dir,
            layer,
            file_name
        )
        if os.path.exists(output_file):
            continue
        
            
        os.makedirs(os.path.join(output_dir, layer), exist_ok=True)

        savemat(output_file, dict([('feat', np.mean(out_text_list, 0, keepdims=True)),
            ('raw_feat', out_text_list)]) )
        
            
# %%

# %%%
if __name__ == '__main__':

    extract_NSD_CLIP_text_features()
