<div align="center">   
  
# Fast Point Transformer
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2007.00151-green)](https://arxiv.org/abs/2112.04702)

</div>

This repository contains the source code and data for our paper:

[Fast Point Transformer](https://arxiv.org/abs/2112.04702) \
 [Chunghyun Park](https://github.com/chrockey),
 [Yoonwoo Jeong](https://github.com/jeongyw12382),
 [Minsu Cho](http://cvlab.postech.ac.kr/~mcho/),
 [Jaesik Park](http://jaesik.info/) \
 POSTECH GSAI & CSE \
 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022

 <div style="text-align:center">
 <img src="assets/overview.png" alt="An overview of the proposed method"/>
 </div>

 ## Overview

 This paper introduces Fast Point Transformer that consists of a new lightweight self-attention layer. Our approach encodes continuous 3D coordinates, and the voxel hashing-based architecture boosts computational efficiency. The proposed method is demonstrated with 3D semantic segmentation and 3D detection. The accuracy of our approach is competitive to the best voxel based method, and our network achieves 129 times faster inference time than the state-of-the-art, Point Transformer, with a reasonable accuracy trade-off in 3D semantic segmentation on S3DIS dataset.

## Installation
This repository is developed and tested on

- Ubuntu 18.04 and 20.04
- Conda 4.11.0
- CUDA 11.1
- Python 3.8.13
- PyTorch 1.7.1 and 1.10.0
- MinkowskiEngine 0.5.4

### Environment Setup
One can install the environment by using the provided shell script (`setup.sh`):
```bash
bash setup.sh fpt
```

## Citation
If you find our code or paper useful, please consider citing our paper:

 ```BibTeX
@inproceedings{park2022fast,
  title={{Fast Point Transformer}},
  author={Chunghyun Park and Yoonwoo Jeong and Minsu Cho and Jaesik Park},
  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledment

Our code is based on the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [Torch-Points3D](https://github.com/torch-points3d/torch-points3d).
