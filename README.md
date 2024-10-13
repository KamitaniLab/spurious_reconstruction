# Spurious-reconstruction

This repository contains demo code for the paper:  
Ken Shirakawa, Yoshihiro Nagano, Misato Tanaka, Shuntaro C. Aoki, Yusuke Muraki, Kei Majima, and Yukiyasu Kamitani, "Spurious reconstruction from brain activity" ([arXiv](https://arxiv.org/abs/2405.10078)).

## Getting Started

### Installation
To clone this repository on your local machine, use the following `git clone` command with the project URL:


``` 
git clone https://github.com/KamitaniLab/spurious_reconstruction.git
```
### Environment Setup
The environment for this project was created using [Rye](https://rye.astral.sh/). To set it up, follow the official instructions to install Rye. After installation, you can synchronize the environment for this repository using:

```
cd spurious_reconstruction 
rye sync
```

## Usage

---

We are currently preparing to share the preprocessed data and files. These files will be made available on Figshare at [this URL](https://figshare.com/articles/dataset/Spurious_reconstruction_from_brain_activity/27013342).

Reproducing the full analysis requires a large amount of data (approximately 3TB) and some analyses take considerable time to complete (up to ~2 weeks). Therefore, we recommend downloading only the minimum necessary files to reproduce specific analysis, such as image reconstruction, UMAP results or simulation analysis.

For example, if you want to reproduce the UMAP results (related to Figure 4), you can download the required data by running the following command:

```
python ./download.py "UMAP visualization analysis"
```

Alternatively, you can directly download the results of UMAP analysis with:
```
python ./download.py "UMAP visualization results"
```
Note: At the moment, only the download of **result data** is supported. Please wait a little longer for the availability of **analysis data**.

Each analysis directory contains additional information on how to reproduce the corresponding results. Please check the README files within each directory for specific instructions.


### Original Dataset
We used publicly available datasets for this project. The key datasets are:

- **NSD (Natural Scenes Dataset)**  
    - Raw fMRI data (including visual images and text annotations): Available upon request via [Spurious Reconstruction@figshare](https://forms.gle/eT4jHxaWwYUDEf2i9).
  
- **Deep Image Reconstruction**  
    - Raw fMRI data: Available from [OpenNeuro](https://openneuro.org/datasets/ds001506).
    - Preprocessed fMRI data, DNN features extracted from images, and decoded DNN features: Available via [Deep Image Reconstruction@figshare](https://github.com/KamitaniLab/DeepImageReconstruction?tab=readme-ov-file#:~:text=Preprocessed%20fMRI%20data,Image%20Reconstruction%40figshare).
    - Visual images: Available upon request via [this form](https://forms.gle/ujvA34948Xg49jdn9).
    - Text annotations: Available from [GOD Stimuli Annotations](https://github.com/KamitaniLab/GOD_stimuli_annotations).

If you want to fully reproduce the analysis, you can use these brain data, image stimuli and corresponding text annotations.
