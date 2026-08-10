
# Feature decoding (feature translation) analysis

This directory contains scripts for feature decoding (feature translation). The decoded (translated) features can be used for image reconstruction (Figure 2 in the main text) and for performance evaluation (Figures 5 and 6 in the main text).

Since completing this task can take a significant amount of time (1 to 6 days), we recommend using the download script to obtain the pre-decoded (translated) features.

## Feature decoding analysis (Related to Figure 2 or Figure 5)

To perform feature decoding analysis on VGG19 features of the Deeprecon dataset, follow these steps:

#### 1. Decoder Training
Run the following command to train the decoder:
```
run python ./analysis/1_case_study/feature-decoding/featdec_fastl2lir_train.py ./analysis/1_case_study/config/deeprecon_fmriprep_rep5_500voxel_caffe_VGG19_allunits_fastl2lir_alpha100.yaml
```
#### 2. Decoder Testing
After training, test the decoder using this command:
```
python ./analysis/1_case_study/feature-decoding/featdec_fastl2lir_predict.py ./analysis/1_case_study/config/deeprecon_fmriprep_rep5_500voxel_caffe_VGG19_allunits_fastl2lir_alpha100.yaml
```
Alternatively, you can skip this time-consuming process and directly get decoded features by running download script:
```
python download.py "image reconstruction analysis"
```

#### 3. Evaluation (Figure 5)
To reproduce the zero-shot identification analysis in Figure 5, you may need to download files, especially you download the decoded features.
```
python download.py "zero-shot analysis"
```
This downloads the true and the decoded VGG19 features of the NSD stimuli and the true
Versatile Diffusion features (approximately 242 GB in total).

Then, the zero-shot identification can be performed by:
```
python ./analysis/1_case_study/feature-decoding/featdec_eval_zero-shot_identification.py <configuration file>
```
Note: this script also reads the fMRI data listed under `training fmri` in the configuration
file, which is not part of the Figshare deposit, and the decoded Versatile Diffusion features
have not been deposited either. Only the VGG19 half of Figure 5 can currently be reproduced
from the downloaded files.
Since this analysis also takes much time and needs heavy resources, you can download the results:
```
python download.py "zero-shot identification results"
```
The figure can be reproduced by 
```
python ./analysis/1_case_study/feature-decoding/Figure_zero_shot_sample_identification.py
```

## Hold out analysis (Figure 6)

#### 1. Decoder training 
```rye run python ./analysis/1_case_study/feature-decoding/featdec_cv_fastl2lir_train.py ./analysis/1_case_study/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml``` 
#### 2. Decoder test
```rye run python ./analysis/1_case_study/feature-decoding/featdec_cv_fastl2lir_predict.py ./analysis/1_case_study/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml``` 

Note: the cross-validated decoders and their decoded features are not distributed, because the
trained decoders of a single split are about 569 GB. Please run the two steps above to
reproduce them. If you only need the numbers behind Figure 6, download the precomputed results
with `python download.py "hold-out results"`.
#### 3. Evaluation (Related to Figure 6)
Evaluate the performance for cluster identification using:
```
python ./analysis/1_case_study/feature-decoding/featdec_cv_eval_cluster_identification.py ./analysis/1_case_study/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml
```
For pairwise identification, run:
```
python ./analysis/1_case_study/feature-decoding/featdec_cv_eval_pairwise_identification.py ./analysis/1_case_study/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml
```
The figures can be get by:
```
python ./analysis/1_case_study/feature-decoding/Figure_hold_out_analysis_pairwise_identification.py
python ./analysis/1_case_study/feature-decoding/Figure_hold_out_analysis_cluster_identification.py
```
---

Ensure that the required data files are prepared and the environment is properly set up before executing these commands.

