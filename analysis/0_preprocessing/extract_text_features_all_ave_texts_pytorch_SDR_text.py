from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import clip
import numpy as np
import os
import sys
from glob import glob

import pandas as pd
import sys
sys.path.append('nsd_access/')
from nsd_access import NSDAccess

from scipy.io import savemat
import sys

# Settings -------------------------------------------------------------------
nsda = NSDAccess('/home/nu/data/fmri_shared/public/NSD/latest/')
# GPU usage settings
device = 'cuda:0'

cache_dir = './data/HF_cache/'
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16, cache_dir=cache_dir)
pipe = pipe.to(device)
network ='stable_diffusion_v1_4' 
#network = 'ViT-B/32'
network_name = network.replace('/','_').replace('-', '_')

output_base_dir = '/home/nu/data/contents_shared/NSD-stimuli/derivatives/'
output_dir = os.path.join(output_base_dir, 'features', 'pytorch', network_name)
os.makedirs(output_dir, exist_ok=True)

# Load network

total_num = 73000
for img in range(total_num):
    ci = nsda.read_image_coco_info([img], info_type='captions', show_annot=True)
    #print(img)
    out_text_list  =[]
    for annotator in ci:
        cap = annotator['caption']
        print(cap)
       
        prompt = None
        image=  None
        strength=  0.8
        num_inference_steps= 50
        guidance_scale = 7.5
        negative_prompt= None
        num_images_per_prompt = 1
        eta= 0.0
        generator = None
        prompt_embeds = None
        negative_prompt_embeds = None
        output_type= "pil"
        return_dict= True
        callback= None
        callback_steps= 1
        # setting
        prompt = [ cap]
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        
        #text  = clip.tokenize(cap).to(device)
        out_text = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        

        
        out_text_list.append(out_text.cpu().detach().numpy())
    #assert 1 ==0
    out_text_list = np.array(out_text_list)
    save_dir = os.path.join(output_dir, 'all_ave_text_encoder')
    # create directory
    os.makedirs(save_dir,exist_ok=True)
    # set file name
    #file_name =os.path.splitext(os.path.basename(imgf))[0] + '.mat'
    file_name = f'nsd{img+1:05}'+ '.mat'
    print( np.mean(out_text_list, 0)[np.newaxis].shape)
    if os.path.isfile(os.path.join(save_dir,file_name)) == False:
        savemat(os.path.join(save_dir,file_name), dict([('feat', np.mean(out_text_list, 0)[np.newaxis]),
                                            
                                                                        ('raw_feat', out_text_list)]) )
       
    else:
        print('the model_output was already existed, skipped')
        #assert 1==0
