Scripts for image reconstruction (Figures 2 and 7 in the main text).
This repository only support iCNN reconstruction methods (Shen et al., 2019). If you want to reconsturct from the other methods, please use the author's official repository:
- [StableDiffusionReconstruction (Takagi and Nishimoto, 2023)](https://github.com/yu-takagi/StableDiffusionReconstruction)
- [Brain-Diffuser (Ozcelik and VanRullen, 2023)](https://github.com/ozcelikfu/brain-diffuser)


To run the reconstruction scripts, you need to prepare the several files, including decoded (tranlated) feature files of the test data, and model parameters. You can download those files by running download scripts.


# Usage (using rye environment)
If you want to reconstruct ImageNetTest data from their decoded features, running this command:

```rye run python ./analysis/1_case_study/image-reconstruction/iCNN/recon_bdpy_icnn_image_AdamW_dist.py  ./analysis/1_case_study/image-reconstruction/iCNN/configs/recon_icnn_ImageNetTest_vgg19_relu7generator_AdamW_1000iter_decoded.yaml``` 

If you want to reconstruct NSD test data from their decoded features, running this command:

```rye run python ./analysis/1_case_study/image-reconstruction/iCNN/recon_bdpy_icnn_image_AdamW_dist.py  ./analysis/1_case_study/image-reconstruction/iCNN/config/recon_icnn_nsd_vgg19_relu7generator_AdamW_1000iterl_decoded.yaml``` 


If you want to reconstruct ArtificialShapes data from their true features, running this command:

```rye run python ./analysis/1_case_study/image-reconstruction/iCNN/recovery_check_bdpy_icnn_image_AdamW_dist.py  ./analysis/1_case_study/image-reconstruction/iCNN/config/recon_icnn_ArtificialShapes_vgg19_relu7generator_AdamW_1000iter_decoded.yaml``` 