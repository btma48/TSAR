# OSAT
Code for "One Step at A Time: A Two-Stage Approach to Motion Artifact Reduction in OCTA Images"
<hr />

> **Abstract:** *Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, has shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).* 
<hr />

## Network Architecture

<img src = "https://i.imgur.com/ulLoEig.png"> 

## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run OSAT.

## Results
Experiments are performed for artifact removal. 

<details>
<summary><strong>Image Deraining</strong> (click to expand) </summary>

<img src = "https://i.imgur.com/mMoqYJi.png"> 
</details>
