Scripts for feature decoding (feature translation). The results of decoded (translated) features are further used for image reconstruction (Figure 2 in the main text) or performance evaluation (Figures 5 and 6 in the main text).

Since completing this task takes time (1~6 days), we reccomend to run the download script to get the decoded (translated) features.

# Usage (using rye environment)

## Feature decoding (related to Figure 2 or 5)
If you want to feature decoding analysis of VGG19 features of the Deeprecon dataset, running this command:

Decoder training 
```rye run python ./analysis/1_case_study/feature-decoding/featdec_fastl2lir_train.py ./analysis/1_case_study/config/deeprecon_fmriprep_rep5_500voxel_caffe_VGG19_allunits_fastl2lir_alpha100.yaml``` 
Decoder test
```rye run python ./analysis/1_case_study/feature-decoding/featdec_fastl2lir_predict.py ./analysis/1_case_study/config/deeprecon_fmriprep_rep5_500voxel_caffe_VGG19_allunits_fastl2lir_alpha100.yaml``` 

Evaluation (Figure 5) 


## Hold-out analysis (related to Figure 6)

Decoder training 
```rye run python ./analysis/1_case_study/feature-decoding/featdec_cv_fastl2lir_train.py ./analysis/1_case_study/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml``` 
Decoder test
```rye run python ./analysis/1_case_study/feature-decoding/featdec_cv_fastl2lir_predict.py ./analysis/1_case_study/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml``` 

Evaluation (Related to Figure 6) 
```rye run python ./analysis/1_case_study/feature-decoding/featdec_cv_eval_cluster_identification.py ./analysis/1_case_study/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml``` 

```rye run python ./analysis/1_case_study/feature-decoding/featdec_cv_eval_pairwise_identification.py ./analysis/1_case_study/config/umap_space_holdout_split_cv_nsd-betasfithrfGLMdenoiseRR_testshared1000_trainnoave_testave_fastl2lir_alpha_100000_versatile_diffusion.yaml``` 