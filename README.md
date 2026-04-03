# StPedf
StPedf: Cell Trajectory Inference of Spatial Transcriptomics via Spatial Proximity Embedding and Spatial Density-adaptive Fusion
StPedf is a Python framework for inferring cellular trajectories and vector fields from spatial transcriptomics data. It integrates graph-based spatial embedding, adaptive transition matrix construction, pseudotime estimation, and velocity field computation to uncover dynamic cellular processes in tissue architecture.
<img width="1512" height="1134" alt="论文框架图最新_01" src="https://github.com/user-attachments/assets/19f094b8-6803-4041-91ec-08fd81ec0860" />
## 📦 Installation

### 1. Create and activate a conda environment

We recommend using Python 3.9:

```bash
conda create -n StPedf python=3.9
conda activate StPedf
2. Install PyTorch
StPedf requires PyTorch 2.0.1 with CUDA 11.8 (or CPU version).
Visit PyTorch official website for your system configuration.

For CUDA 11.8:

bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
For CPU-only:

bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
3. Install other dependencies
Download requirements.txt from this repository and run:

bash
pip install -r requirements.txt
The requirements.txt includes:

text
anndata==0.9.1
scanpy==1.9.3
pandas==2.2.3
numpy==1.26.4
scipy==1.11.4
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
igraph==0.11.8
python-igraph==0.11.8
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
sphinx_gallery
pygam==0.10.1
statsmodels==0.14.6
ipywidgets==8.1.8
POT==0.9.6
gudhi==3.11.0
4. Install StPedf
Clone the repository and install in editable mode:

bash
git clone https://github.com/yourusername/StPedf.git
cd StPedf
pip install -e .
📂 Data preparation
The repository includes example datasets under the data/ folder:

Simulated_dataset/: Synthetic data with known ground truth (linear and bifurcating trajectories)

axolotl brain regeneration: D15 time point after injury

ICC: Intrahepatic cholangiocarcinoma primary tumor

DLPFC: Human dorsolateral prefrontal cortex (Visium)

To run the example notebooks, ensure the data paths match the file structure.

🚀 Usage example
The main example notebook is example.ipynb which demonstrates a complete analysis pipeline:

Load spatial transcriptomics data (simulated dataset)

Preprocess and select highly variable genes

Construct spatial graph

Train StPedf model

Build adaptive transition matrix

Infer pseudotime

Compute vector field and visualize trajectories

Evaluate against ground truth

You can also explore other notebooks:

1.axolotl_brain_injury_D15_trajectory_analysis.ipynb – Axolotl brain regeneration

2.icc2.ipynb – ICC tumor analysis

3.DLPFC151673.ipynb – Human cortical layers
