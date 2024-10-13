# %%
import argparse
import os
from glob import glob
import shutil
import bdpy
from bdpy.util import makedir_ifnot
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import PIL
from PIL import Image, ImageDraw, ImageFont
def draw_group_image_set(condition_list, background_color = (255, 255, 255), 
                         image_size = (160, 160), image_margin = (1, 1, 0, 0), group_margin = (20, 0, 20, 0), max_column_size = 13, 
                         title_fontsize = 20, title_top_padding = 70, title_left_padding = 15,
                         id_show = False, id_fontcolor = "black", id_fontsize = 18, image_id_list = [], maintain_aspect_ratio=False,
                         image_padding_color=(0, 0, 0)):
    """
    condition_list : list
        Each condition is a dictionary-type object that contains the following information:
        ```
            condition = {
                "title" : string, # Title name
                "title_fontcolor" :  string or list,   # HTML color name or RGB value list 
                "image_list": list, # The list of image filepath, ndarray or PIL.Image object.  
            }
        ```
        You can also use "image_filepath_list" instead of "image_list".
    background_color : list or tuple
        RGB value list like [Red, Green, Blue].
    image_size: list or tuple
        The image size like [Height, Width].
    image_margin: list or tuple
        The margin of an image like [Top, Right, Bottom, Left].
    group_margin : list or tuple
        The margin of the multiple row images as [Top, Right, Bottom, Left].
    max_column_size : int
        Maximum number of images arranged horizontally.
    title_fontsize : int
        The font size of titles.
    title_top_padding : 
        Top margin of the title letter.
    title_left_padding : 
        Left margin of the title letter.
    id_show : bool
        Specifying whether to display id name.
    id_fontcolor : list or tuple
        Font color of id name.
    id_fontsize : int
        Font size of id name.
    image_id_list : list
        List of id names.
        This list is required when `id_show` is True.
    """

    #------------------------------------
    # Setting
    #------------------------------------

    for condition in condition_list:
        if not condition.get("image_filepath_list") and not condition.get("image_list"):
            raise RuntimeError("The element of `condition_list` needs `image_filepath_list` or `image_list`.")
            return;
        elif condition.get("image_filepath_list") and not condition.get("image_list"):
            condition["image_list"] = condition["image_filepath_list"]

    total_image_size = len(condition_list[0]["image_list"])
    column_size = np.min([max_column_size, total_image_size]) 

    # create canvas
    turn_num = int(np.ceil(total_image_size / float(column_size)))
    nImg_row = len(condition_list) * turn_num 
    nImg_col = 1 + column_size # 1 means title column 
    size_x = (image_size[0] + image_margin[0] + image_margin[2]) * nImg_row + (group_margin[0] + group_margin[2]) * turn_num
    size_y = (image_size[1] + image_margin[1] + image_margin[3]) * nImg_col + (group_margin[1] + group_margin[3])
    image = np.ones([size_x, size_y, 3])
    for bi, bc in enumerate(background_color):
        image[:, :, bi] = bc

    #------------------------------------
    # Draw image
    #------------------------------------
    for cind, condition in enumerate(condition_list):
        title = condition['title']
        image_list = condition['image_list']

        for tind in range(total_image_size):
            # Load image
            an_image = image_list[tind]
            if an_image is None: # skip
                continue;
            elif isinstance(an_image, str): # str: filepath
                image_obj = Image.open(an_image)
            elif isinstance(an_image, np.ndarray): # np.ndarray: array
                image_obj = Image.fromarray(an_image)
            elif hasattr(an_image, "im"): # im attribute: PIL.Image
                image_obj = an_image
            else:
                raise RuntimeError("What can be treated as an element of `image_list` is only str, ndarray or PIL.Image type.")
                return

            image_obj = image_obj.convert("RGB")
            if maintain_aspect_ratio:
                image_obj = expand2square(image_obj, image_padding_color)
            image_obj = image_obj.resize((image_size[0], image_size[1]), Image.LANCZOS)

            # Calc image position
            row_index = cind + (tind // column_size) * len(condition_list) 
            column_index = 1 + tind % column_size
            turn_index = tind // column_size       
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            y = image_margin[3] + group_margin[3] + column_index * (image_size[1] + image_margin[1] + image_margin[3]) 
            image[ x:(x+image_size[0]), y:(y+image_size[1]), : ] = np.array(image_obj)[:,:,:]

    #------------------------------------
    # Prepare for drawing text
    #------------------------------------
    # cast to unsigned int8
    image = image.astype('uint8')

    # convert ndarray to image object
    image_obj = Image.fromarray(image)
    draw = ImageDraw.Draw(image_obj)

    #------------------------------------
    # Draw title name 
    #------------------------------------
    # Use default font instead of loading from a path
    draw.font = ImageFont.load_default()
    for cind, condition in enumerate(condition_list):
        for turn_index in range(turn_num):
            # Calc text position
            row_index = cind + turn_index * len(condition_list) 
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            x += title_top_padding 
            y = title_left_padding

            # textの座標指定はxとyが逆転するので注意
            if "title_fontcolor" not in condition.keys():
                title_fontcolor = "black"
            else:
                title_fontcolor = condition["title_fontcolor"]
            draw.text([y, x], condition["title"], title_fontcolor)

    #------------------------------------
    # Draw image id name 
    # * image_id_list variables is necessary
    #------------------------------------

    if id_show:
        draw.font = ImageFont.load_default()
        for tind in range(total_image_size):
            #  Calc text position
            row_index = (tind // column_size) * len(condition_list) 
            column_index = 1 + tind % column_size
            turn_index = tind // column_size            
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            x -= id_fontsize
            y = image_margin[3] + group_margin[3] + column_index * (image_size[1] + image_margin[1] + image_margin[3]) 

            draw.text([y, x], image_id_list[tind], id_fontcolor)
            
    return image_obj


# %%
save_dir = './results_dl/assets/fig02'
os.makedirs(save_dir, exist_ok=True)


# NSD
print("NSD")
true_image_dir = './data/NSD-stimuli/source/'
true_image_ext = 'png'
recon_image_ext = 'tiff'

true_image_all = list(np.sort((glob(os.path.join(true_image_dir, '*.' + true_image_ext)))))

selected_images =  np.sort(['nsd03050', 'nsd03435', 'nsd05302', 'nsd07008', 'nsd10065', 'nsd22264',
                    'nsd37802' , 'nsd19201', 'nsd26436',  'nsd10007',  'nsd21509',  'nsd67803'
                   ])
true_images = [true_image for true_image in true_image_all  if os.path.basename(true_image).split('.')[0] in selected_images]
print(len(true_images))

recon_path_dict = {
    
   #"StableDiffusionReconstruction": "./results/reconstruction/NSD-stimuli/derivatives/reconstruction/TN_SD_v1_4/img2img/decoded/gen_5_image2imagensd-betasfithrfGLMdenoiseRR_trainnoave_testave_fastl2lir_a1_nsd-betasfithrfGLMdenoiseRR_trainnoave_testave_fastl2lir_a1/nsd-01/streams_early-streams_ventralgen_0/",
    
    #"BrainDiffuser": "./results/reconstruction/NSD-stimuli/derivatives/reconstruction/...
    
    
    "iCNN": "./results_dl/reconstruction/NSD-stimuli/derivatives/reconstruction/icnn/recon_bdpy_icnn_image_gd_dist_vgg19_relu7generator_gd_scaling_feature_std_train_mean_center_1000iter/decoded/NSD-stimuli/decoded_features"
    }


image_set = [
            {'title': '', 'image_filepath_list': true_images},
        ]

image_path_list = []
for i, (cond_name, recon_path) in enumerate(recon_path_dict.items()):
    print(cond_name)

    recon_images = glob(f'{recon_path}/*.tiff')
    recon_images = [recon_image for recon_image in recon_images  if os.path.basename(recon_image).split('.')[0].split('/')[-1] in selected_images]
    recon_images =list(np.sort(recon_images)) 
    print(len(recon_images))
    image_set.append(
            {'title': "",#f'{cond_name}',
             'image_filepath_list': recon_images},
        )
    #assert 1== 0
    
# %%
img = draw_group_image_set(
                image_set,
                max_column_size=12,
               title_left_padding = 0,
           title_fontsize = 18,
              
            )

img.save(f'{save_dir}/reconstruction_NSD_selected.pdf')

    
# %%
# Deeprecon
# reported image selection
print("Deeprecon")
# %%
true_image_dir = './data/ImageNetTest/source/'
true_image_ext = 'JPEG'
recon_image_ext = 'tiff'

true_image_all = list(np.sort((glob(os.path.join(true_image_dir, '*.' + true_image_ext)))))


selected_images = [ "n01443537_22563", "n01858441_11077","n02139199_10398", "n02690373_7713", "n03710193_22225", "n04252077_10859"]

true_images = [true_image for true_image in true_image_all  if os.path.basename(true_image).split('.')[0] in selected_images ]
print(len(true_images))


recon_path_dict = {
    
    #"StableDiffusionReconstruction": "./results/reconstruction/ImageNetTest/derivatives/reconstruction/TN_SD_v1_4/img2img/decoded/gen_5_image2imagedeeprecon-fmriprep_trainnoave_testave_fastl2lir_no_vox_select_a1_deeprecon-fmriprep_trainnoave_testave_fastl2lir_no_vox_select_a1/dr-01/VC-VCgen_0/",
    
    #"Brain-Diffuser": "./results/reconstruction/ImageNetTest/derivatives/reconstruction/versatile_diffusion/vd_doublecond_scaled_pytorch/decoded/deeprecon_testImageNet_trainnoave_testave_fastl2lir_alpha_100000/dr-01/VC/",
    
      "iCNN": "./results/reconstruction/ImageNetTest/derivatives/reconstruction/icnn/recon_bdpy_icnn_image_AdamW_dist_vgg19_relu7generator_gd_scaling_feature_std_train_mean_center_800iter/decoded/ImageNetTest/decoded_features"
  
    }
# %%
# save all images 

image_set = [
            {'title': '', 'image_filepath_list': true_images},
        ]

image_path_list = []
for i, (cond_name, recon_path) in enumerate(recon_path_dict.items()):
    print(cond_name)

    recon_images = glob(f'{recon_path}/*.tiff')
    recon_images = [recon_image for recon_image in recon_images  if os.path.basename(recon_image).split('.')[0].split('/')[-1] in selected_images]
    recon_images =list(np.sort(recon_images)) 
    print(len(recon_images))
    image_set.append(
            {'title': "",#f'{cond_name}',
             'image_filepath_list': recon_images},
        )

    
    

# 画像をプロットする
# %%
img = draw_group_image_set(
                image_set,
                max_column_size=6,
               title_left_padding = 0,
           title_fontsize = 18,
              
            )

img.save(f'{save_dir}/reconstruction_Deeprecon_selected.pdf')


# ArtificialShapes
print("ArtificialShapes")
true_image_dir = './data/ArtificialShapes/source/'
true_image_ext = 'tiff'
recon_image_ext = 'tiff'

true_image_all = list(np.sort((glob(os.path.join(true_image_dir, '*.' + true_image_ext)))))

selected_images =  np.sort(["colorExpStim01_red_square", 
                            "colorExpStim03_red_largering", 
                            "colorExpStim04_red_+", 
                            "colorExpStim09_green_+", 
                            "colorExpStim31_white_square","colorExpStim40_black_X"])

true_images = [true_image for true_image in true_image_all  if os.path.basename(true_image).split('.')[0] in selected_images]
print(len(true_images))

recon_path_dict = {
    
   #"StableDiffusionReconstruction": "./results/reconstruction/NSD-stimuli/derivatives/reconstruction/TN_SD_v1_4/img2img/decoded/gen_5_image2imagensd-betasfithrfGLMdenoiseRR_trainnoave_testave_fastl2lir_a1_nsd-betasfithrfGLMdenoiseRR_trainnoave_testave_fastl2lir_a1/nsd-01/streams_early-streams_ventralgen_0/",
    
    #"BrainDiffuser": "./results/reconstruction/NSD-stimuli/derivatives/reconstruction/...
    
    
    "iCNN": "./results/reconstruction/ArtificialShapes/derivatives/reconstruction/icnn/recon_bdpy_icnn_image_gd_dist_vgg19_relu7generator_gd_scaling_feature_std_train_mean_center_1000iter/decoded/ArtificialShapes/decoded_features"
    }


image_set = [
            {'title': '', 'image_filepath_list': true_images},
        ]

image_path_list = []
for i, (cond_name, recon_path) in enumerate(recon_path_dict.items()):
    print(cond_name)

    recon_images = glob(f'{recon_path}/*.tiff')
    recon_images = [recon_image for recon_image in recon_images  if os.path.basename(recon_image).split('.')[0].split('/')[-1] in selected_images]
    recon_images =list(np.sort(recon_images)) 
    print(len(recon_images))
    image_set.append(
            {'title': "",#f'{cond_name}',
             'image_filepath_list': recon_images},
        )
    
# %%
img = draw_group_image_set(
                image_set,
                max_column_size=6,
               title_left_padding = 0,
           title_fontsize = 18,
              
            )

img.save(f'{save_dir}/reconstruction_ArtificialShapes_selected.pdf')

# %%
