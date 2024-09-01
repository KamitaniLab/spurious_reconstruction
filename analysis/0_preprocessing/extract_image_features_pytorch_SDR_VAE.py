
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

from PIL import Image
import numpy as np
import PIL
import torch
import clip
import numpy as np
import os
import sys
from glob import glob

import pandas as pd


from scipy.io import savemat


PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]).resize((320,320), resample=PIL_INTERPOLATION["lanczos"])
                         )[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

# GPU usage settings

device = 'cuda:0'

image_dir = '/home/nu/data/contents_shared/NSD-stimuli/source'
image_ext = 'png'


cache_dir = '/home/nu/ken.shirakawa/HF_cache/'
device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16, cache_dir=cache_dir)
pipe = pipe.to(device)
network ='stable_diffusion_v1_4' #<- seems to be used in Stable diffusion (it is large)
#network = 'ViT-B/32'
network_name = network.replace('/','_').replace('-', '_')
#output_dir = os.path.join('/home/nu/data/contents_shared/NSD-stimuli/derivatives/features/default', 'pytorch', network_name)
output_base_dir = '/home/nu/data/contents_shared/NSD-stimuli/derivatives/'
output_dir = os.path.join(output_base_dir, 'features', 'pytorch', network_name)

image_output_dir = os.path.join(output_base_dir, 'dec_image', 'pytorch', network_name)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)




# Extract features
image_files = glob(os.path.join(image_dir, '*.' + image_ext))
for imgf in image_files:
    #SD setting
    num_inference_steps = 50
    strength = 0.8
    batch_size = 1
    num_images_per_prompt = 1
    
    print('Image:  %s' % imgf)

    # Open an image
    img =Image.open(imgf).convert('RGB')
    
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
        
    # 4. Preprocess image
    prep_image = preprocess(img)
    
    # 5. set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    
    # 6. Prepare latent variables
    #latents = pipe.prepare_latents(
    #        prep_image, latent_timestep, batch_size, num_images_per_prompt, torch.float16, device, generator=None
    #    )
    prep_image = prep_image.to(device=device, dtype=torch.float16)

    _batch_size = batch_size * num_images_per_prompt
    init_latents = pipe.vae.encode(prep_image).latent_dist.sample(None)
    init_latents = pipe.vae.config.scaling_factor * init_latents
    latents = torch.cat([init_latents], dim=0)
    
    #save 
    save_dir = os.path.join(output_dir, 'vae_latent')
    # create directory
    os.makedirs(save_dir,exist_ok=True)
    # set file name
    file_name =os.path.splitext(os.path.basename(imgf))[0] + '.mat'
    latents_numpy = latents.cpu().detach().numpy()
    if os.path.isfile(os.path.join(save_dir,file_name)) == False:
        savemat(os.path.join(save_dir,file_name), dict([('feat', latents_numpy)]) )
    else:
        print('the model_output was already existed, skipped')
        #assert 1==0

    
    
    # 9. Post-processing
    with torch.no_grad():
        dec_image= pipe.decode_latents(latents)
        
    dec_image = pipe.numpy_to_pil(dec_image)
    save_dir = os.path.join(image_output_dir, 'vae_recon')
    os.makedirs(save_dir,exist_ok=True)
    file_name =os.path.splitext(os.path.basename(imgf))[0] + '.png'
    if os.path.isfile(os.path.join(save_dir,file_name)) == False:
        try:
            dec_image[0].save(os.path.join(save_dir,file_name))
        except:
            dec_image.save(os.path.join(save_dir,file_name))
        #savemat(os.path.join(save_dir,file_name), dict([('feat', model_output)]) )
    else:
        print('the vae_recon was already existed, skipped')
        #assert 1==0


