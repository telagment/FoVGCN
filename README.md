# FoVGCN

PyTorch implementation of paper [[AN EFFECTIVE FOVEATED 360° IMAGE ASSESSMENT BASED ON GRAPH CONVOLUTION NETWORK]] "https://ieeexplore.ieee.org/abstract/document/9878309"


## Install
- pip install Pillow==6.2.0
- pip install opencv_python==4.1.0.25
- pip install scipy==1.2.1
- pip install torch==1.1.0 torchvision==0.3.0

## Prepare Data
- Download [database]: "https://drive.google.com/drive/folders/1dt6pz3WUy5D2JYrdiiVvkCu-55R7bcG1?usp=sharing"

## Training and testing
- cd FoVGCN
- CUDA_LAUNCH_BLOCKING=1 python 'main_multicases.py' --save test

## Citation

Huong, Truong Thu, et al. "An Effective Foveated 360° Image Assessment Based on Graph Convolution Network." IEEE Access 10 (2022): 98165-98178.
