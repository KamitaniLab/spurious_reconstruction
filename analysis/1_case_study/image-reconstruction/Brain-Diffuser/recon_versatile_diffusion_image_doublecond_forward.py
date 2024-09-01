'''iCNN reconstruction; gradient descent, with image generator'''

import sys
import argparse
import glob
from itertools import product
import os

import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import scipy.io as sio
import yaml
import torch
import torchvision.transforms as tvtrans

from bdpy.dataform import Features, DecodedFeatures

# Main function ##############################################################

def recon_vdvae_image_decoder(
        features_dir,
        recon_output_dir='./recon',
        recon_image_ext='tiff',
        recon_targets=None,
        seed_image_dir='',
        seed_image_prefix='',
        seed_image_ext='tiff',
        vd_local_path = "../versatile_diffusion/",
        subjects=None, rois=None,
        seed=0,
        device='cuda:0'
):
    '''
    - VDVAE reconstruction by forwaring decoded features
    - Decoder: VDVAE
    '''

    # Network settings -------------------------------------------------------
    sys.path.insert(0, vd_local_path)
    # Settings for versatile 
    from lib.cfg_helper import model_cfg_bank
    from lib.model_zoo import get_model
    from lib.model_zoo.ddim_vd import DDIMSampler_VD
    from lib.model_zoo.vd import VD

    # Reconstruction options -------------------------------------------------

    # Main #######################################################################
    # Setup results directory ------------------------------------------------
    if not os.path.exists(recon_output_dir):
        os.makedirs(recon_output_dir)

    # Set reconstruction options ---------------------------------------------
    upsampled_image_size = [512, 512]
    strength = 0.75 # これに50をかけると37になる．すなわちiteration回数．
    mixing = 0.4
    ddim_steps = 50
    ddim_eta = 0
    scale = 7.5
    cfgm_name = 'vd_noema'
    sampler = DDIMSampler_VD
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    cfgm = model_cfg_bank()(cfgm_name) # この関数が相対パスで直下に"versatile_diffusion"ディレクトリを持っていることを前提としている． 

    # Load model
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)    

    # Send CLIP model and VAE model to GPU
    net.clip.to(device)
    net.autokl.to(device)

    # Set float16
    net.autokl.half()

    # Create sampler 
    sampler = sampler(net)

    # Set seed 
    torch.manual_seed(seed)

    # Reconstrucion ----------------------------------------------------------
    for subject, roi in product(subjects, rois):
        decoded = subject is not None and roi is not None
        
        print('----------------------------------------')
        if decoded:
            print('Decoded image reconstuction')
            print('Subject: ' + subject)
            print('ROI:     ' + roi)
        else:
            print('True image reconstuction')
        print('')
        
        if decoded:
            save_dir = os.path.join(recon_output_dir, subject, roi)
        else:
            save_dir = os.path.join(recon_output_dir)
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get images if images is None
        if recon_targets is None:
            if decoded:
                matfiles = glob.glob(os.path.join(features_dir, 'text_encoder', subject, roi, '*.mat'))
            else:
                matfiles = glob.glob(os.path.join(features_dir, 'text_encoder', '*.mat'))
            images = [os.path.splitext(os.path.basename(fl))[0] for fl in matfiles]
        else:
            images = recon_targets[:]
        print("Recon target stimuli top10:", images[:10])
        
        # Load DNN features
        if decoded:
            features = DecodedFeatures(os.path.join(features_dir), squeeze=False)
        else:
            features = Features(features_dir)

        # Images loop
        for image_label in images:
            print('Image: ' + image_label)

            # Districuted computation control
            snapshots_dir = os.path.join(save_dir, 'snapshots', 'image-%s' % image_label)
            if os.path.exists(snapshots_dir):
                print('Already done or running. Skipped.')
                #continue
            else:
                os.makedirs(snapshots_dir)

            # Load DNN latent features
            if decoded:
                text_feat = features.get(layer='text_encoder', subject=subject, roi=roi, label=image_label)[0]
                image_feat = features.get(layer='vision_encoder', subject=subject, roi=roi, label=image_label)[0]
                seed_image = PIL.Image.open(os.path.join(seed_image_dir, subject, roi, seed_image_prefix + image_label + "." + seed_image_ext))
            else:
                text_feat = features.get(layer='text_encoder', label=image_label)[0]
                image_feat = features.get(layer='vision_encoder', label=image_label)[0]
                seed_image = PIL.Image.open(os.path.join(seed_image_dir, seed_image_prefix + image_label + "." + seed_image_ext))

            text_feat = torch.tensor(text_feat[np.newaxis], dtype=torch.half, requires_grad=False).to(device)
            image_feat = torch.tensor(image_feat[np.newaxis], dtype=torch.half, requires_grad=False).to(device)
                
            # Convert seed image to VAE latent feature
            seed_image = regularize_image(seed_image, upsampled_image_size)
            seed_image = seed_image*2 - 1
            seed_image = seed_image.unsqueeze(0).to(device).half()
            init_latent = net.autokl_encode(seed_image)
            
            # Prepare unconditional features
            dummy_text = ''
            uncond_text_feat = net.clip_encode_text([dummy_text])
            uncond_text_feat = uncond_text_feat.to(device).half()
            dummy_image = torch.zeros((1, 3, 224, 224)).to(device)
            uncond_image_feat = net.clip_encode_vision(dummy_image)
            uncond_image_feat = uncond_image_feat.to(device).half()

            # Make scheduler
            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=True)
            # Fix the iteration number from 50 to 37!!!
            t_enc = int(strength * ddim_steps)

            # Forward process
            # Add noise to latent feature
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
            z_enc = z_enc.to(device)
                
            # Send sampler to GPU
            sampler.model.model.diffusion_model.device=device
            sampler.model.model.diffusion_model.half().to(device)          
            
            # Reverse process
            # Reduce noise from the latent feature based on predicted vision/text CLIP features
            z = sampler.decode_dc(
                x_latent=z_enc,
                first_conditioning=[uncond_image_feat, image_feat], 
                second_conditioning=[uncond_text_feat, text_feat], 
                t_start=t_enc,
                unconditional_guidance_scale=scale,
                xtype='image', 
                first_ctype='vision',
                second_ctype='prompt',
                mixed_ratio=(1-mixing), 
                )
            
            # Reverse the VAE latent feature to image
            z = z.to(device).half()
            x = net.autokl_decode(z)
            
            # Adjust image
            x_clamp = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0) # 1 x 3 x 512 x 512
            recon_img = tvtrans.ToPILImage()(x_clamp[0])
            #recon_img
            
            # Save the recon image array and latent features
            recon_image_mat_file = os.path.join(save_dir, 'recon_image' + '-' + image_label + '.mat')
            sio.savemat(recon_image_mat_file, {'recon_image': np.array(recon_img), 'z': z.detach().cpu().numpy()})
            # Save the image
            recon_image_normalized_file = os.path.join(save_dir, 'recon_image' + '-' + image_label + '.' + recon_image_ext)
            recon_img.save(recon_image_normalized_file)
            
    print('All done')

    return recon_output_dir


# Functions ################################################################

def regularize_image(x, upsampled_image_size=[512, 512]):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = PIL.Image.open(x).resize(upsampled_image_size, resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize(upsampled_image_size, resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize(upsampled_image_size, resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'

    assert (x.shape[1]==upsampled_image_size[0]) & (x.shape[2]==upsampled_image_size[1]), \
        'Wrong image size'
    return x

# Entry point ################################################################

if __name__ == '__main__':
    # # For debug
    # sys.argv = ["recon.py", "test_image_wise_no_overlap_config/recon_nsd-betasfithrfGLMdenoiseRR_nsd03050_no_train_overlap_trainnoave_testave_fastl2lir_a100000_versatile_diffusion.yaml"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    args = parser.parse_args()

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    with open(conf['feature decoding'], 'r') as f:
        conf_featdec = yaml.safe_load(f)

    conf.update({
        'feature decoding': conf_featdec
    })

    if 'analysis name' in conf['feature decoding']:
        analysis_name = conf['feature decoding']['analysis name']
    else:
        analysis_name = ''
    
    if 'recon targets' in conf:
        recon_targets = conf['recon targets']
    else:
        recon_targets = None

    if 'scaled decoded feature' in conf and conf['scaled decoded feature'] is True:
        features_dir = os.path.join(
            conf['feature decoding']['decoded feature dir'],
            analysis_name,
            'scaled_decoded_features',
            conf['feature decoding']['network']
        )
    else:
        features_dir = os.path.join(
            conf['feature decoding']['decoded feature dir'],
            analysis_name,
            'decoded_features',
            conf['feature decoding']['network']
        )
    
    recon_vdvae_image_decoder(
        features_dir = features_dir,
        recon_output_dir=os.path.join(conf['recon output dir'], analysis_name),
        recon_image_ext=conf['recon image ext'],
        recon_targets=recon_targets,
        seed_image_dir=conf['seed image dir'],
        seed_image_prefix=conf['seed image prefix'],
        seed_image_ext=conf['seed image ext'],
        vd_local_path=conf['vd local path'],
        subjects=conf['recon subjects'],
        rois=conf['recon rois'],
        seed=conf[seed'],
        device='cuda:0'
    )
