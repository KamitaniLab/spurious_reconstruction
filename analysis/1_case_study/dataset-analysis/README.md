Scripts for performing UMAP visualization (Figure 4 in the main text).

To run the reconstruction scripts, you need to prepare the several files, including extracted feature files. You can download those files by running download scripts.

# Usage (using rye environment)
If you want to project CLIP text feature of NSD into UMAP embedding space, running this command:

```rye run python ./analysis/1_case_study/dataset-analysis/umap_transform_nsd_text_features.py``` 

The visualization results can be seen from the jupyter files:

```Show_umap_plot.ipynb```

[Optional]
You can also see the point corresponding to the image or text annoatation interactively:
```Interactive_umap_plot.ipynb```