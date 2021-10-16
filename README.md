# ImageTransfer-RTUCON2020

This vault is to share the code used in paper 'Image Transfer Applied in Electric Machine Optimization' presented in RTUCON2020

The main coding language used in this project is Python (to control deep-learning solver) and Matlab (to control finite-element solver & optimization)

The software used for data generation is JMAG

The deep learning framework used is PyTorch and the main adapted model is [Pix2Pix](https://phillipi.github.io/pix2pix/) from Phillip Isola

Feel free to download, study, modify and share this project.

if you have any question, please contact me on 13816901408@163.com

### Sichao Yang


## Environment

1. 创建虚拟环境 (python>=3.7)

```
conda create -n imageT python=3.7
```

2. 安装支持cuda11.1的[pytorch](https://pytorch.org/get-started/previous-versions/)

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```


