# Image Reconstruction Analysis

This directory contains scripts for performing image reconstruction, corresponding to Figures 2 and 7 in the main text. The repository currently supports only iCNN reconstruction methods (Shen et al., 2019). If you want to use other reconstruction methods, please refer to the authors' official repositories:

- [StableDiffusionReconstruction (Takagi and Nishimoto, 2023)](https://github.com/yu-takagi/StableDiffusionReconstruction)
- [Brain-Diffuser (Ozcelik and VanRullen, 2023)](https://github.com/ozcelikfu/brain-diffuser)

## Prerequisites

To run the reconstruction scripts, you need to prepare sesveral files, including DNN features and corresponding decoded (translated) features, and model parameters beforehand. You can download those files by running download scripts.
```
python ./download.py "image reconstruction analysis"
```

This downloads the following files (approximately 62 GB in total) into `./data`:

| File | Contents | Size |
|:--|:--|--:|
| `data-models_shared.zip` | VGG19 encoder and caffenet relu7 generator parameters | 4.4 GB |
| `data-ImageNetTest-derivatives-decoded_features-...-caffe-VGG_ILSVRC_19_layers.zip` | Decoded VGG19 features of the Deeprecon test stimuli (Figure 2) | 2.5 GB |
| `data-ImageNetTest-derivatives-features-caffe-VGG_ILSVRC_19_layers.zip` | True VGG19 features of the Deeprecon test stimuli | 2.7 GB |
| `data-ArtificialShapes-derivatives-decoded_features-...-caffe-VGG_ILSVRC_19_layers.zip` | Decoded VGG19 features of the artificial shapes (Figure 7) | 2.0 GB |
| `data-ArtificialShapes-derivatives-features-caffe-VGG_ILSVRC_19_layers.zip` | True VGG19 features of the artificial shapes | 0.7 GB |
| `data-NSD-stimuli-derivatives-decoded_features-...-caffe-VGG_ILSVRC_19_layers-<layer>.zip` (7 files) | Decoded VGG19 features of the NSD test stimuli (Figure 2) | 49.4 GB |

Note: the reconstruction itself runs with these files, but comparing the reconstructions
with the presented images (`evaluation.true_image`, i.e. `./data/<dataset>/source`) requires
the visual stimuli, which are available upon request only. See the top-level README.

Optionaly, you can prepraed decoded feature, conducting feature decoding analysis at `feature-decoding directory`.

### Usage

If you want to reconstruct Deeprecon data from their decoded features, running this command:

```
 python ./analysis/1_case_study/image-reconstruction/iCNN/recon_bdpy_icnn_image_AdamW_dist.py ./analysis/1_case_study/image-reconstruction/iCNN/configs/recon_icnn_ImageNetTest_vgg19_relu7generator_AdamW_1000iter_decoded.yaml
```
If you want to reconstruct NSD test data from their decoded features, running this command:

```
python ./analysis/1_case_study/image-reconstruction/iCNN/recon_bdpy_icnn_image_AdamW_dist.py  ./analysis/1_case_study/image-reconstruction/iCNN/config/recon_icnn_nsd_vgg19_relu7generator_AdamW_1000iterl_decoded.yaml
``` 
The similar figure related to Figure 2 in the paper can be produced by:
```
python ./analysis/1_case_study/image-reconstruction/Figure_image_reconstruction.py

```

The similar figure related to Figure 7 in the paper can be obtrained by:
Get recovered images from true features. For example, if you want to reconstruct ArtificialShapes data from their true features, running this command:

```python ./analysis/1_case_study/image-reconstruction/iCNN/recovery_check_bdpy_icnn_image_AdamW_dist.py  ./analysis/1_case_study/image-reconstruction/iCNN/config/recon_icnn_ArtificialShapes_vgg19_relu7generator_AdamW_1000iter_decoded.yaml
```

Then running:

```
python ./analysis/1_case_study/image-reconstruction/Figure_image_recovery_check.py

```
---

Ensure that all required data files and environment configurations are in place before running the commands.