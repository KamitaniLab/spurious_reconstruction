'''SD model image generation from true/decoded CLIP features'''

import bdpy
import yaml
import sys
from itertools import product
sys.path.insert(1,'/home/nu/ken.shirakawa/projects/python_KS/python/bdpy/')

from bdpy.dataform import Features, DecodedFeatures

from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import hdf5storage
import argparse, os, sys, glob
#import cv2
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm, trange



def recon_SD_image2image(
        image_features_dir,
        text_features_dir,
        
        img2img_output_dir='./recon',
        zonly_output_dir='./z_only_recon',
        subjects=None, image_rois=None, text_rois=None,
        image_feature_train_mean=None,
        image_feature_train_std=None,
        text_feature_train_mean=None,
        text_feature_train_std=None,
        scale_factor= None,
        selected_labels = [],
        device='cuda:0',
        gen_N=5
    ):
    
    # load stable diffusion model
    cache_dir = './data/HF_cache/'
    model_id = "CompVis/ldm-text2im-large-256"
    model_path = "CompVis/stable-diffusion-v1-4"

    #pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    #   model_path,
    #  torch_dtype=torch.float16, cache_dir=cache_dir)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16, cache_dir=cache_dir)
    pipe = pipe.to(device)
    
    
    #feat_mean0_train = feature_train_mean
    #
    image_layers = ['vae_latent']
    text_layers = ['text_encoder']
        
    for subject, image_roi, text_roi in product(subjects, image_rois, text_rois):
        
        #decoded = subject is not None and roi is not None
        decoded = subject != "None" and image_roi !=  "None" and text_roi !=  "None"

        print('----------------------------------------')
        if decoded:
            print('Subject: ' + subject)
            print('Image ROI: ' + image_roi)
            print('Text ROI: ' + text_roi)
        print('')

        if decoded:
            save_dir = os.path.join(img2img_output_dir, subject, f'{image_roi}-{text_roi}')
            save_dir_ = os.path.join(zonly_output_dir, subject, f'{image_roi}-{text_roi}')
        else:
            save_dir = os.path.join(img2img_output_dir)
            save_dir_ = os.path.join(zonly_output_dir)
        #if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir_, exist_ok=True)
        
        # Get images if images is None
        if decoded:
            image_matfiles = glob.glob(os.path.join(image_features_dir, image_layers[0], subject, image_roi, '*.mat'))
            text_matfiles = glob.glob(os.path.join(text_features_dir, text_layers[0], subject, text_roi, '*.mat'))
        else:
            image_matfiles = glob.glob(os.path.join(image_features_dir, image_layers[0], '*.mat'))
            text_matfiles = glob.glob(os.path.join(text_features_dir, image_layers[0], '*.mat'))
        images = [os.path.splitext(os.path.basename(fl))[0] for fl in image_matfiles]
        
        if selected_labels != []:
            images = [fl for fl in images if fl in selected_labels]
        
        # Load DNN features
        if decoded:
            image_features = DecodedFeatures(os.path.join(image_features_dir), squeeze=False)
            text_features = DecodedFeatures(os.path.join(text_features_dir), squeeze=False)
        else:
            image_features = Features(image_features_dir)
            text_features = Features(text_features_dir)
            
        # Images loop
        for jj, image_label in enumerate(images):
            print('Image: ' + image_label)
            #assert 1== 0
            n_save_dir_ = save_dir+ f'gen_{0}'
            if os.path.exists(os.path.join(n_save_dir_,'recon_image_normalized' + '-' + image_label + '.tiff')):
                print('Already done. Skipped.')
                continue


            # Load DNN features
            if decoded:
                try:
                    image_feat = {
                        layer: image_features.get(layer=layer, subject=subject, roi=image_roi, label=image_label)
                        for layer in image_layers
                    }
                    
                    text_feat = {
                        layer: text_features.get(layer=layer, subject=subject, roi=text_roi, label=image_label)
                        for layer in text_layers
                    }
                except:
                    image_feat = {
                        layer: image_features.get(layer=layer_mapping[layer], subject=subject, roi=roi, label=image_label)
                        for layer in image_layers
                    }
                    
                    text_feat = {
                        layer: text_features.get(layer=layer_mapping[layer], subject=subject, roi=roi, label=image_label)
                        for layer in text_layers
                    }
            else:
                labels = image_features.labels
                
                image_feat = {
                    layer: image_features.get(layer, label=image_label)
                    for layer in image_layers
                }
                
                text_feat = {
                    layer: text_features.get(layer, label=image_label)
                    for layer in text_layers
                }
               
            #----------------------------------------
            # Normalization of decoded features
            #----------------------------------------
            for layer, ft in image_feat.items():

                if decoded: 
                    assert ft.shape[0] == 1
                    # negative_prompts = np.load('SD_CLIP_negative_prompt.npy')
                    #first_768 = np.load('SD_CLIP_first_768_dim.npy')

                    #ft[0][0] = negative_prompts
                    #ft[0][1][0] = first_768
                    try:
                        ft = (ft - image_feature_train_mean[layer]) * scale_factor + image_feature_train_mean[layer] 
                    except:
                        ft = (ft - image_feature_train_mean[layer_mapping[layer]]) * scale_factor + image_feature_train_mean[layer_mapping[layer]] 

                    #ft[0][0] = negative_prompts
                    #ft[0][1][0] = first_768 


                image_feat.update({layer: ft})
            target_image_feat_dict_torch = {
                                        layer: image_feat[layer]
                                        for layer in image_layers
                                    }   
            # decode images
            latents = torch.tensor(target_image_feat_dict_torch['vae_latent'].astype(np.float16)).to(device)
            with torch.no_grad():
                X_z_tensor= pipe.decode_latents(latents)
            #a size of 320 ⇥ 320, and then resized it to [512 ⇥ 512.
            X_z = pipe.numpy_to_pil(X_z_tensor)[0].resize([512,512])
            recon_image_normalized_file = os.path.join(save_dir_, 'recon_image_normalized' + '-' + image_label + '.tiff')
            X_z.save(recon_image_normalized_file)
            
            # image 2 image
            for layer, ft in text_feat.items():

                if decoded: 
                    assert ft.shape[0] == 1
                    negative_prompts = np.load('SD_CLIP_negative_prompt.npy')
                    first_768 = np.load('SD_CLIP_first_768_dim.npy')

                    #ft[0][0] = negative_prompts
                    #ft[0][1][0] = first_768
                    try:
                        ft = (ft - text_feature_train_mean[layer]) * scale_factor + text_feature_train_mean[layer] 
                    except:
                        ft = (ft - text_feature_train_mean[layer_mapping[layer]]) * scale_factor + text_feature_train_mean[layer_mapping[layer]] 

                    ft[0][0] = negative_prompts
                    ft[0][1][0] = first_768 


                text_feat.update({layer: ft})
            target_text_feat_dict_torch = {
                                        layer: text_feat[layer]
                                        for layer in text_layers
                                    }
            
            target_text_feat = target_text_feat_dict_torch['text_encoder']
            for n in range(gen_N):
                n_save_dir = save_dir+ f'gen_{n}'
                os.makedirs(n_save_dir, exist_ok=True)
                # create black pil images
                recon_images = Image.new('RGB', (512, 512), (0, 0, 0))
                while True:
                 
                    pipe_output  = pipe(image=X_z, prompt_embeds = torch.tensor(target_text_feat.astype(np.float16)).to(device).squeeze(), guidance_scale = 5.0)
                    recon_images = pipe_output.images[1]
                    # if recon_images are all black images then regenerate
                    if np.mean(np.array(recon_images)[0]) != 0:
                        break
                    print('Image are black rerun!')
                        
                                                        
                recon_image_normalized_file = os.path.join(n_save_dir, 'recon_image_normalized' + '-' + image_label + '.tiff')
                recon_images.save(recon_image_normalized_file)
                
            
        

# Entry point ################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',

    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='the number of device',

    )
    parser.add_argument(
        '--gen_N',
        type=int,
        default=5,
        help='the number z_c_images',

    )
    args = parser.parse_args()

    conf_file = args.conf
    device_num = args.device
    device = torch.device(f'cuda:{device_num}')

    N = args.gen_N

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    if 'image feature decoding' in conf:
        with open(conf['image feature decoding'], 'r') as f:
            image_conf_featdec = yaml.safe_load(f)

        conf.update({
            'image feature decoding': image_conf_featdec
        })
    
    if 'text feature decoding' in conf:
        with open(conf['text feature decoding'], 'r') as f:
            text_conf_featdec = yaml.safe_load(f)

        conf.update({
            'text feature decoding': text_conf_featdec
        })

    if 'image feature decoding' in conf:
        if 'analysis name' in conf['image feature decoding']:
            image_analysis_name = conf['image feature decoding']['analysis name']
        else:
            image_analysis_name = ''
    elif 'analysis name' in conf:
        image_analysis_name = conf['analysis name']

    else:
        image_analysis_name = 'true_feat'
        
    if 'text feature decoding' in conf:
        if 'analysis name' in conf['text feature decoding']:
            text_analysis_name = conf['text feature decoding']['analysis name']
        else:
            text_analysis_name = ''
    elif 'analysis name' in conf:
        text_analysis_name = conf['analysis name']

    else:
        text_analysis_name = 'true_feat'
        
    analysis_name = f'{image_analysis_name}_{text_analysis_name}'
        
    if 'image feature decoding' in conf:
        image_feature_path =os.path.join(
                conf['image feature decoding']['decoded feature dir'],
                image_analysis_name,
                'decoded_features',
                conf['image feature decoding']['network']
            )
        layer_analyzed = conf['image feature decoding']['layers']
        one_subject = list(conf['image feature decoding']['test fmri'].keys())[0]
        one_roi = list(conf['image feature decoding']['rois'].keys())[0]
        # load feature_train_mean_file
        image_feature_train_mean ={}
        image_feature_train_std ={}
        for layer in layer_analyzed:
            mean_file_path = os.path.join(
                                        conf['image feature decoding']['feature decoder dir'],
                                        conf['image feature decoding']['analysis name'],
                                        conf['image feature decoding']['network'],
                                        layer, 
                                        one_subject, 
                                        one_roi,
                                        'model',
                                        'y_mean.mat')

            mean_val = hdf5storage.loadmat(mean_file_path)['y_mean']
            image_feature_train_mean[layer] = mean_val
            
            
            norm_file_path = os.path.join(
                                        conf['image feature decoding']['feature decoder dir'],
                                        conf['image feature decoding']['analysis name'],
                                        conf['image feature decoding']['network'],
                                        layer, 
                                        one_subject, 
                                        one_roi,
                                        'model',
                                        'y_norm.mat')

            std_val = hdf5storage.loadmat(norm_file_path)['y_norm']
            image_feature_train_std[layer] = std_val

    else:
        image_feature_path = conf['true image feature']
        image_feature_train_mean  =None
        image_feature_train_std  =None
        
        
    if 'text feature decoding' in conf:
        text_feature_path =os.path.join(
                conf['text feature decoding']['decoded feature dir'],
                text_analysis_name,
                'decoded_features',
                conf['text feature decoding']['network']
            )
        layer_analyzed = conf['text feature decoding']['layers']
        one_subject = list(conf['text feature decoding']['test fmri'].keys())[0]
        one_roi = list(conf['text feature decoding']['rois'].keys())[0]
        # load feature_train_mean_file
        text_feature_train_mean ={}
        text_feature_train_std ={}
        for layer in layer_analyzed:
            mean_file_path = os.path.join(
                                        conf['text feature decoding']['feature decoder dir'],
                                        conf['text feature decoding']['analysis name'],
                                        conf['text feature decoding']['network'],
                                        layer, 
                                        one_subject, 
                                        one_roi,
                                        'model',
                                        'y_mean.mat')

            mean_val = hdf5storage.loadmat(mean_file_path)['y_mean']
            text_feature_train_mean[layer] = mean_val
            
            
            norm_file_path = os.path.join(
                                        conf['text feature decoding']['feature decoder dir'],
                                        conf['text feature decoding']['analysis name'],
                                        conf['text feature decoding']['network'],
                                        layer, 
                                        one_subject, 
                                        one_roi,
                                        'model',
                                        'y_norm.mat')

            std_val = hdf5storage.loadmat(norm_file_path)['y_norm']
            text_feature_train_std[layer] = std_val

    else:
        text_feature_path = conf['true text feature']
        text_feature_train_mean  =None
        text_feature_train_std  =None
        
    if 'include test label' in conf:
        included_labels = conf['include test label']
    else:
        included_labels = []

    recon_SD_image2image(
        image_feature_path,
        text_feature_path,
        img2img_output_dir=os.path.join(conf['recon output dir'], f'gen_{N}_image2image' + analysis_name),
        zonly_output_dir=os.path.join(conf['recon output dir'], f'gen_{N}_z_only' + analysis_name),
        subjects=conf['recon subjects'],
        image_rois=conf['recon image rois'],
        text_rois=conf['recon text rois'],
        scale_factor= conf['scale factor'],
        image_feature_train_mean=image_feature_train_mean,
        image_feature_train_std=image_feature_train_std,
        text_feature_train_mean=text_feature_train_mean,
        text_feature_train_std=text_feature_train_std,
        selected_labels = included_labels, 
        device=f'cuda:{device_num}',
        gen_N=N
    )