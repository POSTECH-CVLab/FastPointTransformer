# Fast Point Transformer
### [Project Page](http://cvlab.postech.ac.kr/research/FPT/) | [Paper](https://arxiv.org/abs/2112.04702)

[Fast Point Transformer](https://arxiv.org/abs/2112.04702)  
 [Chunghyun Park](https://chrockey.github.io/),
 [Yoonwoo Jeong](https://yoonwoojeong.medium.com/about),
 [Minsu Cho](http://cvlab.postech.ac.kr/~mcho/),
 [Jaesik Park](http://jaesik.info/)<br>
 POSTECH GSAI & CSE<br>
in CVPR 2022

<div style="text-align:center">
<img src="assets/overview.png" alt="An Overview of the proposed pipeline"/>
</div>

## Overview
This work introduces Fast Point Transformer that consists of a new lightweight self-attention layer. Our approach encodes continuous 3D coordinates, and the voxel hashing-based architecture boosts computational efficiency. The proposed method is demonstrated with 3D semantic segmentation and 3D detection. The accuracy of our approach is competitive to the best voxel based method, and our network achieves 129 times faster inference time than the state-of-the-art, Point Transformer, with a reasonable accuracy trade-off in 3D semantic segmentation on S3DIS dataset.

## Citation
If you find our code or paper useful, please consider citing our paper:

 ```BibTeX
@inproceedings{park2022fast,
  title={{Fast Point Transformer}},
  author={Chunghyun Park and Yoonwoo Jeong and Minsu Cho and Jaesik Park},
  booktitle={Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## Experiments
### 1. S3DIS Area 5 test
We denote MinkowskiNet42 trained with this repository as MinkowskiNet42<sup>&dagger;</sup>.
We use voxel size 4cm for both MinkowskiNet42<sup>&dagger;</sup> and our Fast Point Transformer.
We highlight the **best** method in the table below.

| Model                             | Latency (sec) | mAcc (%) | mIoU (%) | Reference |
|:----------------------------------|--------------------:|:--------:|:--------:|:---------:|
| PointTransformer                  | 18.07 | 76.5 | 70.4 | [Codes from authors](https://github.com/POSTECH-CVLab/point-transformer) |
| MinkowskiNet42<sup>&dagger;</sup> | **0.08**  | 74.1 | 67.2 | [checkpoint] |
| &nbsp;&nbsp;+ rotation average    | 0.66  | 75.1 | 69.0 | - |
| FastPointTransformer              | 0.14 | 76.6 | 69.2 | [checkpoint] |
| &nbsp;&nbsp;+ rotation average    | 1.13  | **77.6** | **71.0** | - |

### 2. ScanNet V2 validation
#### 2-1. 3D semantic segmentation
| Model                             | Voxel Size  | mAcc (%) | mIoU (%) | Reference |
|:----------------------------------|:-----------:|:--------:|:--------:|:---------:|
| MinkowskiNet42                    | 2cm | - | 72.2 | [Official GitHub](https://github.com/chrischoy/SpatioTemporalSegmentation) |
| MinkowskiNet42<sup>&dagger;</sup> | 2cm | 81.4 | 72.1 | [checkpoint] |
| FastPointTransformer              | 2cm | 81.2 | 72.5 | [checkpoint] |
| MinkowskiNet42<sup>&dagger;</sup> | 5cm | 76.3 | 67.0 | [checkpoint] |
| FastPointTransformer              | 5cm | 78.9 | 70.0 | [checkpoint] |
| MinkowskiNet42<sup>&dagger;</sup> | 10cm | 70.8 | 60.7 | [checkpoint] |
| FastPointTransformer              | 10cm | 76.1 | 66.5 | [checkpoint] |

#### 2-2. 3D object detection
| Model                             | mAP@0.25 | mAP@0.5 | Reference |
|:---------------------------------:|:--------:|:-------:|:---------:|
| MinkowskiNet42<sup>&dagger;</sup> |  |  | [checkpoint] |
| FastPointTransformer              |  |  | [checkpoint] |

## Installation
This repository is developed and tested on

- Ubuntu 18.04 and 20.04
- Conda 4.11.0
- CUDA 11.1
- Python 3.8.13
- PyTorch 1.7.1 and 1.10.0
- MinkowskiEngine 0.5.4

### Environment Setup
One can install the environment by using the provided shell script:
```bash
~$ git clone --recursive git@github.com:chrockey/FastPointTransformer.git
~$ cd FastPointTransformer
~/FastPointTransformer$ bash setup.sh fpt
```

## Acknowledment

Our code is based on the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [Torch-Points3D](https://github.com/torch-points3d/torch-points3d).
We also thank [Hengshuang Zhao](https://hszhao.github.io/) for providing [the code](https://github.com/POSTECH-CVLab/point-transformer) of [Point Transformer](https://arxiv.org/abs/2012.09164).
If you use our model, please consider citing them as well.
