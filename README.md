# StPedf
StPedf: Cell Trajectory Inference of Spatial Transcriptomics via Spatial Proximity Embedding and Spatial Density-adaptive Fusion

StPedf is a Python framework for inferring cellular trajectories and vector fields from spatial transcriptomics data. It integrates graph-based spatial embedding, adaptive transition matrix construction, pseudotime estimation, and velocity field computation to uncover dynamic cellular processes in tissue architecture.
<img width="1512" height="944" alt="论文框架图最新_01(1)" src="https://github.com/user-attachments/assets/f186fc07-a57b-456e-a455-132ae9a7539b" />



StPedf is a framework for spatial transcriptomics trajectory analysis. This repository provides installation instructions, example datasets, and notebooks demonstrating the complete analysis workflow.

## 📦 Installation

### 1. Create and activate a conda environment

We recommend using Python 3.9:

```bash
conda create -n StPedf python=3.9
conda activate StPedf
 ```
### 2. Install PyTorch

StPedf requires PyTorch 2.0.1 with CUDA 11.8 (or a CPU-only version).

Please visit the official PyTorch website
 to select the appropriate installation command for your system.

For CUDA 11.8:
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
 ```
For CPU-only:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```
### 3. Install other dependencies

Download requirements.txt from this repository and run:
```bash
pip install -r requirements.txt
```
The requirements.txt includes:
```bash
anndata==0.9.1
scanpy==1.9.3
pandas==2.2.3
numpy==1.26.4
scipy==1.11.4
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
igraph==0.11.8
harmonypy==0.0.10
umap-learn==0.5.7
squidpy==1.6.1
spatialdata==0.2.5
gseapy==1.1.9
plotly==6.1.0
tqdm==4.65.0
pillow==11.2.1
scikit-image==0.25.2
h5py==3.13.0
nbsphinx==0.9.3
sphinx==7.2.6
sphinx-hoverxref==1.3.0
sphinx-rtd-theme==3.1.0
sphinx-gallery
pygam==0.10.1
statsmodels==0.14.6
ipywidgets==8.1.8
POT==0.9.6
gudhi==3.11.0
```
## 📂 Data Preparation

The repository currently includes the following local dataset:

```text
StPedf/
└── Simulated_dataset/
```
Due to file size limitations, the real datasets are not included in this repository. Please download them separately before running the corresponding notebooks.

1. Axolotl brain regeneration and DLPFC datasets

These datasets are provided via Baidu Netdisk:

Folder name: StPedf_data
Link: https://pan.baidu.com/s/1oOV0vwgRD8gvk9BohfER1A?pwd=z556

2. ICC dataset

The primary tumor spatial transcriptomics (ST) data for intrahepatic cholangiocarcinoma (ICC) can be downloaded from the CNGBdb database.

Accession number: CNP0002199

## 🚀 Usage Example

The main example notebook is `example.ipynb`, which demonstrates a complete analysis pipeline:

1. Load spatial transcriptomics data (simulated dataset)
2. Preprocess the data and select highly variable genes
3. Construct the spatial graph
4. Train the StPedf model
5. Build the adaptive transition matrix
6. Infer pseudotime
7. Compute the vector field and visualize trajectories
8. Evaluate results against the ground truth

You can also explore other notebooks:

1. `1.axolotl_brain_injury_D15_trajectory_analysis.ipynb` — Axolotl brain regeneration
2. `2.icc2.ipynb` — ICC tumor analysis
3. `3.DLPFC151673.ipynb` — Human cortical layers
