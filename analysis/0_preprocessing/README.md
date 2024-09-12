Scripts for extracting deep neural networks (DNNs) features from images (or text annotation), located at `./data/[dataset]/source` folders. You should put the stimuli files appropriately beforehand or run the download script.

# Usage (using rye environment)
If you want to extract VGG19 (for iCNN) in the Deeprecon dataset, running this command:
```rye run python ./analysis/0_preprocessing/iCNN_image_vgg19_feature_extraction.py  ./analysis/0_preprocessing/config/Deeprecon/ImageNetTest.yaml``` 

If you want to extract CLIP text (for Brain-Diffuser) in the Deeprecon dataset, running this command:
```rye run python ./analysis/0_preprocessing/BD_extract_NSD_CLIP_tex_feature.py  ./analysis/0_preprocessing/config/NSD/NSD.yaml``` 

If you want to extract CLIP vision (for Brain-Diffuser) in the NSD dataset, running this command:
```rye run python ./analysis/0_preprocessing/BD_extract_CLIP_vision_feature.py  ./analysis/0_preprocessing/config/NSD/NSD.yaml``` 

Note:
If you want to extract CLIP features used in the Brain-Diffuser models, you need additonaly prepare[versatile_diffusion](https://github.com/ozcelikfu/brain-diffuser/tree/main/versatile_diffusion) at the top of this directory. 

