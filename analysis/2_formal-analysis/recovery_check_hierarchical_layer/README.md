# Recovery check analysis of each hierarchical layer 

This directory contains the recovery check analysis at each hierarchical layer of deep neural networks (DNNs) in Figure 11 of the paper. You can follow the instructions below and reproduce the figures and results.

## Downloading DNN features

Before running these scripts, you will download the necessary feature files beforehand using the following command:
```
python ./download.py "single layer recovery check analysis"
```
## Three Types of Image Recovery Analysis

We provide three types of image recovery analysis. You can choose from the following methods based on the specific analysis you want to check.

### 1. No Generator (pixel optimization)
This analysis does not use a generator for image reconstruction. Run the following command:
```
python ./analysis/2_formal-analysis/recovery_check_hierarchial_layer/recovery_check_icnn_image_no_generator_true_single_layer.py "./analysis/2_formal-analysis/recovery_check_hierarchial_layer/config/recovery_check_icnn_deeprecon_vgg19_nogenerator_AdamW_800iter_layerwise_image2feat2image.yaml"
```

### 2. Deep Image Prior (DIP) Generator (weak image prior)
This analysis uses a DIP generator as weak image priors. Use this command:
```
python ./analysis/2_formal-analysis/recovery_check_hierarchial_layer/recovery_check_icnn_image_no_generator_true_single_layer.py "./analysis/2_formal-analysis/recovery_check_hierarchial_layer/config/recovery_check_icnn_deeprecon_vgg19_dipgenerator_AdamW_800iter_layerwise_image2feat2image.yaml"
```

### 3. Pretrained Image Generator (pretrained image prior)
This analysis uses a pretrained generator for naturalistic image priors. Run the following command:

```
python ./analysis/2_formal-analysis/recovery_check_hierarchial_layer/recovery_check_icnn_image_pretrained_generator_true_single_layer.py "./analysis/2_formal-analysis/recovery_check_hierarchial_layer/config/recovery_check_icnn_deeprecon_vgg19_relu7generator_AdamW_800iter_layerwise_image2feat2image.yaml"
```


## Downloading Pre-computed Results

If you prefer to skip the time-consuming analysis process, you can download the precomputed results with the following command:
```
python ./download.py "single layer recovery check results"
```
## Reproducing the Figure

Once the results are prepared, you can reproduce the figure by running:
```
python ./analysis/2_formal-analysis/recovery_check_hierarchial_layer/Figure_layerwise_recovery_check.py
```

---

Ensure that the necessary data files are prepared before running these commands.

