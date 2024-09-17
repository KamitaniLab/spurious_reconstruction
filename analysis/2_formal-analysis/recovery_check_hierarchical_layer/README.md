This directory contains recovery check analysis at the each hierarchical layer of DNNs.

Before running these scritps, you will need to download feature files first:

```
python ./download.py  “single layer recovery check analysis”
```

We provide three types of image recovery analysis

### No generator (pixel optimization)
```
python ./analysis/2_formal-analysis/recovery_check_hierarchial_layer/recovery_check_icnn_image_no_generator_true_single_layer.py "./analysis/2_formal-analysis/recovery_check_hierarchial_layer/config/recovery_check_icnn_deeprecon_vgg19_nogenerator_AdamW_800iter_layerwise_image2feat2image.yaml"
```

### Deep Image Prior (DIP) generator (weak image prior)
```
python ./analysis/2_formal-analysis/recovery_check_hierarchial_layer/recovery_check_icnn_image_no_generator_true_single_layer.py "./analysis/2_formal-analysis/recovery_check_hierarchial_layer/config/recovery_check_icnn_deeprecon_vgg19_dipgenerator_AdamW_800iter_layerwise_image2feat2image.yaml"
```

### Pretrained image generator (pretrained image prior)
```
python ./analysis/2_formal-analysis/recovery_check_hierarchial_layer/recovery_check_icnn_image_pretrained_generator_true_single_layer.py "./analysis/2_formal-analysis/recovery_check_hierarchial_layer/config/recovery_check_icnn_deeprecon_vgg19_relu7generator_AdamW_800iter_layerwise_image2feat2image.yaml"
```

Alternatively, you can download the results to reproduce the figures quickly:
```
python ./download.py  “single layer recovery check results”
```

The figure can be created by running:

```
python ./analysis/2_formal-analysis/recovery_check_hierarchial_layer/Figure_layerwise_recovery_check.py
```