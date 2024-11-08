# OSAT
Code for "A Two-Stage Approach to Motion Artifact Reduction in OCTA Images"
<hr />

> **Abstract:** *Optical coherence tomography angiography (OCTA) is an innovative and non-invasive imaging technique that leverages motion contrast imaging to generate angiographic images from high-resolution volumetric blood flow data rapidly. However, OCTA imaging is vulnerable to various artifacts induced by eye movements, including displacement artifacts, duplicated scanning artifacts, and white line artifacts.  These artifacts significantly compromise image quality and impede the widespread adoption of OCTA. Previous attempts to mitigate eye motion artifacts necessitated costly hardware upgrades. However, despite the availability of advanced eye-tracking hardware and software correction in commercial machines, motion artifacts persist in real-world usage. Recently developed cost-effective learning-based methods only addressed individual instances of white line artifacts. The significant challenge of devising a learning-based model capable of effectively handling all three categories of eye motion artifacts remains a substantial challenge that has not been thoroughly explored to date. To address this challenge, we propose a comprehensive two-stage framework, TSAR, to remove three types of eye motion artifacts in OCTA images. In the first stage, we leverage the intrinsic axial and directional attributes of these artifacts in the first phase to develop an innovative hierarchical transformer network. This network is designed to capture global-wise, local-wise, and vertical-wise features effectively while also removing displacement and duplicate scanning artifacts. Afterward, we leverage the contextual information and develop a residual conditional diffusion model (RCDM) to remove the white line artifacts. By applying our TSAR to the degraded OCTA images, we aim to eliminate all three types of motion artifacts. We evaluate the superior performance of our proposed methodology in artifact removal and image quality enhancement compared to other methods by conducting experiments on both synthetic and real-world OCTA images.* 
<hr />

## Network Architecture
The overall framework of  motion artifact removal
<img src = "./sources/framework.PNG"> 

The details architecture of different self-attention modules
<img src = "./sources/self-attention.PNG"> 

The motivation for the proposed self-attention modules
<img src = "./sources/SA.PNG"> 

## Installation

1. Make conda environment
```
conda create -n octa python=3.7
conda activate octa
```

2. Install dependencies
```
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

3. Install basicsr (this project is based on basicsr)
```
python setup.py develop --no_cuda_ext
```

## Run
Config the dataset and model in ./Operations/OCTA.yml, then 
```
bash train.sh
```
Then run the inpainting code in ./rcdm

```
python run.py
```

## Results
Experiments are performed for artifact removal. 

OCTA Motion Artifacts Removal 

<img src = "./sources/result0.PNG"> 


OCTA Motion Artifacts Removal on clinical data
<img src = "./sources/result1.PNG"> 

For the inpainting stage, an example is:

<img src = "./sources/ddpm.PNG"> 


