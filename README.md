# D-SIC: Energy-Efficient Digital Semantic Image Communication via Large Generative Models

This repository contains the code for the paper titled "D-SIC: Energy-Efficient Digital Semantic Image Communication via Large Generative Models". Much of the code in this repository is based on Stable Cascade (https://github.com/Stability-AI/StableCascade).  

## 🌟 Overview
This paper introduces a digital semantic image communication (SIC) framework that incorporates Stable Cascade (SC) to achieve the goal of reliable, efficient and digitally compatible SIC. The architecture of SC is extensively modified to mitigate channel-induced distortions using Channel State Information (CSI) and corrupted image embeddings as conditioning.

## 💻 Installation
* **Step 1: Clone the directory**
  
  Download the code to your local machine and navigate into the project directory.
  ```bash
  git clone https://github.com/abilalk02/D-SIC.git
  
* **Step 2: Set up the Conda environment**

  Create the environment using the provided environment.yml file
  ```bash
  # Create the environment from the provided configuration file
  conda env create -f environment.yml

  # Activate the environment
  conda activate DSIC
  
* **Step 3: Download pretrained model weights**

  The pretrained model weights can be downloaded from https://huggingface.co/khalidr4/DSIC. You can download manually or use the ```huggingface_hub``` Python library. You will be able to download two folders i.e. ```models``` and ```finetuned``` . The first folder contains pretrained weights of the relevant stable cascade models. The second folder contains the pretrained D-SIC model weights for the Cityscapes dataset and the Rician channel (can be used for AWGn and Rayleigh channel as well, with minor degradation in performance). All pretrained model weights for all channels and datasets will eventually be accessible via the huggingface repository.
  ```bash
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download(
      repo_id='khalidr4/DSIC', 
      local_dir='./models', 
      local_dir_use_symlinks=False
  )
  "

* **Step 4: Download training and test datasets**

  Create the environment using the provided environment.yml file
  ```bash
  

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
