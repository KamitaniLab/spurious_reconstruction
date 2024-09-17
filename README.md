# Spurious-reconstruction

Demo code for Ken Shirakawa, Yoshihiro Nagano, Misato Tanaka, Shuntaro C. Aoki, Yusuke Muraki, Kei Majima, and Yukiyasu Kamitani. 

# Getting Started

## Instration
Clone the reposibory on your local machine, using git clone by pasting the URL of this project:

`git clone https://github.com/KamitaniLab/spurious_reconstruction.git`
## Build environment
We created the environment by rye. You can install rye by [the official instruction](https://rye.astral.sh/).
After installing rye, you can sync this repository
```
cd spurious_reconstruction
rye sync
```

## Dataset preparation
We used the public available dataset: 
- NSD
    - Raw fMRI data (including visual images and text annotation): upon request via https://forms.gle/eT4jHxaWwYUDEf2i9
- Deeprecon 
    - Raw fMRI data: [Deep Image Reconstruction@OpenNeuro](https://openneuro.org/datasets/ds001506)
    - Preprocessed fMRI data, DNN features extracted from images, and decoded DNN features: [Deep Image Reconstruction@figshare](https://github.com/KamitaniLab/DeepImageReconstruction?tab=readme-ov-file#:~:text=Preprocessed%20fMRI%20data,Image%20Reconstruction%40figshare)
    - Visual images: upon requeset via https://forms.gle/ujvA34948Xg49jdn9 
    - Text annotation: https://github.com/KamitaniLab/GOD_stimuli_annotations 

You can also use the following commands to download specific data. The data will be automatically extracted and organized into the designated directory:


## Usage
---

We are currently preparing to share the preprocessed data and files. These files will be available at Figshare https://doi.org/10.6084/m9.figshare.27013342


Since reproducing all of the analysis needs a huge data files (~3T storages) and some analyze take long time to finish (~ 2 weeks), we recommend use download scritps to download minimum files to reproduce a specific task such as image reconstruction, simulation analysis etc.. 

If you want to reprodue the UMAP results, you can download the data by:

You can running `download.py` with specific argument such as 
```
python ./download.py "UMAP visualization analysis" # Related to Figure 4
```
Alternatively, you also get the results directly, running by:
```
python ./download.py "UMAP visualization results" # Related to Figure 4
```

Each analysis directory provide more information to reproduce the results.
