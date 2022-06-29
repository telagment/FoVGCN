# FoVGCN

Thanks for your attention. In this repo, we provide the codes for the paper [[AN EFFECTIVE FOVEATED 360° IMAGE ASSESSMENT BASED ON GRAPH CONVOLUTION NETWORK]] "https://www.techrxiv.org/articles/preprint/AN_EFFECTIVE_FOVEATED_360_IMAGE_ASSESSMENT_BASED_ON_GRAPH_CONVOLUTION_NETWORK/19134935"

## Install
- pip install Pillow==6.2.0
- pip install opencv_python==4.1.0.25
- pip install scipy==1.2.1
- pip install torch==1.1.0 torchvision==0.3.0



## Prepare Data
- Download [database]: "https://drive.google.com/drive/folders/1dt6pz3WUy5D2JYrdiiVvkCu-55R7bcG1?usp=sharing"

## Training and testing

- CUDA_LAUNCH_BLOCKING=1 python 'main_multicases.py' --save test

## Citation
You may cite it in your paper. Thanks a lot.

@article{truong2022effective,
  title={AN EFFECTIVE FOVEATED 360° IMAGE ASSESSMENT BASED ON GRAPH CONVOLUTION NETWORK},
  author={Truong, Thu Huong and Do Thu, Ha and Tran, Thi Thanh Huyen and Truong, Thang and Nam, Pham and Nguyen, Thanh and Ngo, Viet and Bui, Tien},
  year={2022},
  publisher={TechRxiv}
}

