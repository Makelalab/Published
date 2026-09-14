# DnaK Dynamics
Repository for the analysis accompanying **“Bacterial Hsp70 DnaK transiently samples the proteome to rapidly capture stress-induced misfolding” (Ada Pajari, Taras Redchuk, and Jarno Mäkelä, 2026)**. It contains Python code and notebooks for single-molecule tracking (SMT) microscopy analysis and figure generation used in the manuscript.


## Features

End-to-end SMT pipeline:
- Spot localization, tracking, and diffusion coefficient estimation (Dapp, log10(Dapp))
- Hidden Markov Model (HMM) state inference
- Gaussian Mixture Model (GMM) clustering
- Heatmaps of localizations, distance-to-aggregate analysis, per cell fluorescence summaries


## Prerequisites
### Python
```
Python: 3.10
Jupyter
pillow
boto3~=1.34.71
botocore~=1.34.71
h5py
ipyfilechooser~=0.6.0
ipywidgets
matplotlib
numpy==1.26.4
pandas
pystackreg~=0.2.7
python-dotenv
s3transfer~=0.10.1
scikit-image==0.22.0
scikit-learn
scipy==1.13.0
seaborn
shap
tifffile==2024.2.12
torch
trackpy~=0.6.2
umap-learn
```
### MatLab
MATLAB R2024b (26.1) + Statistics and Machine Learning Toolbox 26.1 for MATLAB-based steps (Gaussian Mixture Model-analysis


### Citation




