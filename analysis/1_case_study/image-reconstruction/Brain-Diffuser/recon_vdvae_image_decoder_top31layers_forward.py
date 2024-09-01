'''iCNN reconstruction; gradient descent, with image generator'''

import sys
import argparse
import glob
from itertools import product
import os

from bdpy.recon.torch.icnn import reconstruct
from bdpy.recon.utils import normalize_image, clip_extreme
from bdpy.dl.torch.models import layer_map
from bdpy.dataform import Features, DecodedFeatures
from bdpy.feature import normalize_feature
from bdpy.util import dump_info

import numpy as np
import PIL.Image
import scipy.io as sio
import torch
import torch.optim as optim
import yaml


# Main function ##############################################################

def recon_vdvae_image_decoder(
        features_dir,
        recon_output_dir='./recon',
        recon_image_ext='tiff',
        recon_targets=None,
        vdvae_local_path = "../vdvae/",
        subjects=None, rois=None,
        seed=0,
        device='cuda:0'
):
    '''
    - VDVAE reconstruction by forwaring decoded features
    - Decoder: VDVAE
    '''

    # Network settings -------------------------------------------------------
    sys.path.insert(0, vdvae_local_path)
    from data import set_up_data
    #from utils import get_cpu_stats_over_ranks
    from train_helpers import set_up_hyperparams, load_vaes #, load_opt, accumulate_stats, save_model, update_ema
    #from train_helpers import setup_mpi, setup_save_dirs, add_vae_arguments, parse_args_and_update_hparams
    #from hps import Hyperparams, HPARAMS_REGISTRY

    sys.argv = ['',  
                '--hps', 'imagenet64', 
                '--restore_path', os.path.join(vdvae_local_path, 'trained_model/imagenet64-iter-1600000-model.th'), 
                '--restore_ema_path', os.path.join(vdvae_local_path, 'trained_model/imagenet64-iter-1600000-model-ema.th'),
                '--restore_log_path', os.path.join(vdvae_local_path, 'trained_model/imagenet64-iter-1600000-log.jsonl'),
                '--restore_optimizer_path', os.path.join(vdvae_local_path, 'trained_model/imagenet64-iter-1600000-opt.th'),
                '--seed', seed,
                '--test_eval']
    H, logprint = set_up_hyperparams()
    H.data_root = os.path.join(vdvae_local_path, "data")
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H) # preprocess_fnはimagenet64仕様のものが返ってくる
    vae, ema_vae = load_vaes(H, logprint)

    all_layers = ["dec_blocks_%03d" % (i) for i in range(75)] # Use dec_blocks_000 to dec_blocks_074
    target_layers = ["dec_blocks_%03d" % (i) for i in range(31)] # Use dec_blocks_000 to dec_blocks_030

    # Reconstruction options -------------------------------------------------

    # Main #######################################################################
    # Setup results directory ------------------------------------------------
    if not os.path.exists(recon_output_dir):
        os.makedirs(recon_output_dir)

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
                matfiles = glob.glob(os.path.join(features_dir, target_layers[0], subject, roi, '*.mat'))
            else:
                matfiles = glob.glob(os.path.join(features_dir, target_layers[0], '*.mat'))
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
                continue
            else:
                os.makedirs(snapshots_dir)

            # Load DNN latent features
            if decoded:
                labels = features.labels
                latents = []
                for layer in all_layers:
                    if layer in target_layers:
                        latents.append(torch.Tensor(features.get(layer=layer, subject=subject, roi=roi, label=image_label)).to(device=device))
                    else:
                        latents.append(None)
            else:
                labels = features.labels
                latents = []
                for layer in all_layers:
                    if layer in target_layers:
                        latents.append(torch.Tensor(features.get(layer=layer, label=image_label)).to(device=device))
                    else:
                        latents.append(None)

            # Forward latent features
            px_z = ema_vae.decoder.forward_manual_latents(1, latents, t=None)

            # Extract image. 
            # sample関数で0-255の画像にまで変換済み
            recon_image = ema_vae.decoder.out_net.sample(px_z)[0]

            # Save the results
            # Save the raw reconstructed image
            recon_image_mat_file = os.path.join(save_dir, 'recon_image' + '-' + image_label + '.mat')
            print("Save to:", recon_image_mat_file)
            px_z_array = px_z.detach().cpu().numpy()
            sio.savemat(recon_image_mat_file, {'recon_image': recon_image, 'px_z': px_z_array})

            recon_image_normalized_file = os.path.join(save_dir, 'recon_image' + '-' + image_label + '.' + recon_image_ext)
            print("Save to:", recon_image_normalized_file)
            PIL.Image.fromarray(recon_image).save(recon_image_normalized_file)
            
    print('All done')

    return recon_output_dir


# Entry point ################################################################

if __name__ == '__main__':
    # For debug
#    sys.argv = ["recon.py", "test_image_wise_no_overlap_config/recon_nsd-betasfithrfGLMdenoiseRR_nsd03050_no_train_overlap_trainnoave_testave_fastl2lir_a50000_vdvae.yaml"]

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
        vdvae_local_path=conf['vdvae local path'],
        subjects=conf['recon subjects'],
        rois=conf['recon rois'],
        seed=conf['seed'],
        device='cuda:0'
    )
