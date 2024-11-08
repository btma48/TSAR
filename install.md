# Installation

2. Make conda environment
```
conda create -n octa python=3.7
conda activate octa
```

3. Install dependencies
```
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
# our method is based on basicsr
```
python setup.py develop --no_cuda_ext
```

