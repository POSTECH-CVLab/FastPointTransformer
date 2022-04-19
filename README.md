# Fast Point Transformer
### [Project Page](http://cvlab.postech.ac.kr/research/FPT/) | [Paper](https://arxiv.org/abs/2112.04702)
This repository contains the official source code and data for our paper:

>[Fast Point Transformer](https://arxiv.org/abs/2112.04702)  
> [Chunghyun Park](https://chrockey.github.io/),
> [Yoonwoo Jeong](https://yoonwoojeong.medium.com/about),
> [Minsu Cho](http://cvlab.postech.ac.kr/~mcho/), and
> [Jaesik Park](http://jaesik.info/)<br>
> POSTECH GSAI & CSE<br>
> CVPR, 2022, New Orleans.

<div style="text-align:center">
<img src="assets/overview.png" alt="An Overview of the proposed pipeline"/>
</div>

## Overview
This work introduces *Fast Point Transformer* that consists of a new lightweight self-attention layer. Our approach encodes continuous 3D coordinates, and the voxel hashing-based architecture boosts computational efficiency. The proposed method is demonstrated with 3D semantic segmentation and 3D detection. The accuracy of our approach is competitive to the best voxel based method, and our network achieves 129 times faster inference time than the state-of-the-art, Point Transformer, with a reasonable accuracy trade-off in 3D semantic segmentation on S3DIS dataset.

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

| Model                             | Latency (sec) | mAcc (%) | mIoU (%) | Reference |
|:----------------------------------|--------------------:|:--------:|:--------:|:---------:|
| PointTransformer                  | 18.07 | 76.5 | 70.4 | [Codes from the authors](https://github.com/POSTECH-CVLab/point-transformer) |
| MinkowskiNet42<sup>&dagger;</sup> | 0.08  | 74.1 | 67.2 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/EZcO0DH6QeNGgIwGFZsmL-4BAlikmHAHlBs4JBcS5XfpVQ?download=1) |
| &nbsp;&nbsp;+ rotation average    | 0.66  | 75.1 | 69.0 | - |
| FastPointTransformer              | 0.14 | 76.6 | 69.2 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/ER8KwMTzqAxAvK9KeOZ9U_IBuCAuv4hP6zOWD-3HNO6Xeg?download=1) |
| &nbsp;&nbsp;+ rotation average    | 1.13  | 77.6 | 71.0 | - |

### 2. ScanNetV2 validation
| Model                             | Voxel Size  | mAcc (%) | mIoU (%) | Reference |
|:----------------------------------|:-----------:|:--------:|:--------:|:---------:|
| MinkowskiNet42                    | 2cm | - | 72.2 | [Official GitHub](https://github.com/chrischoy/SpatioTemporalSegmentation) |
| MinkowskiNet42<sup>&dagger;</sup> | 2cm | 81.4 | 72.1 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/EXmE1pWDZ8lEtJU7SQMjkXcBnhSMXFTdHWXkMAAF7KeiuA?download=1) |
| FastPointTransformer              | 2cm | 81.2 | 72.5 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/EX_xAyhoNXdJg4eSg2vS_bYB8eFAP7A8FPCYfKOS2T13LQ?download=1) |
| MinkowskiNet42<sup>&dagger;</sup> | 5cm | 76.3 | 67.0 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/EZLG00u5JXJDvOi3sYziOIMB1l6HNN5OW9gTQRFWc6EwzA?download=1) |
| FastPointTransformer              | 5cm | 78.9 | 70.0 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/EXbXclfXZGtMpBZY93zi7M8B_tl8rwM65NK1cumN7QM_8g?download=1) |
| MinkowskiNet42<sup>&dagger;</sup> | 10cm | 70.8 | 60.7 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/EVLn0f5noY1Al6Kos9l-0yABM0qZLFt6d4a3yFgTcQ2Vmw?download=1) |
| FastPointTransformer              | 10cm | 76.1 | 66.5 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/ESO1jLNHO89ApdjguUauqsMBCx_TijA26UOeGbF4XxQwoA?download=1) |

## Installation
This repository is developed and tested on

- Ubuntu 18.04 and 20.04
- Conda 4.11.0
- CUDA 11.1
- Python 3.8.13
- PyTorch 1.7.1 and 1.10.0
- MinkowskiEngine 0.5.4

### Environment Setup
You can install the environment by using the provided shell script:
```bash
~$ git clone --recursive git@github.com:chrockey/FastPointTransformer.git
~$ cd FastPointTransformer
~/FastPointTransformer$ bash setup.sh fpt
~/FastPointTransformer$ conda activate fpt
```

### (3D Semantic Segmentation) Training & Evaluation
First of all, you need to download the datasets (ScanNetV2 and S3DIS), and preprocess them as:
```bash
(fpt) ~/FastPointTransformer$ python src/data/preprocess_scannet.py # you need to modify the data path
(fpt) ~/FastPointTransformer$ python src/data/preprocess_s3dis.py # you need to modify the data path
```
And then, locate the provided meta data of each dataset (`src/data/meta_data`) with the preprocessed dataset following the structure below:

```
${data_dir}
├── scannetv2
│   ├── meta_data
│   │   ├── scannetv2_train.txt
│   │   ├── scannetv2_val.txt
│   │   └── ...
│   └── scannet_processed
│       ├── train
│       │   ├── scene0000_00.ply
│       │   ├── scene0000_01.ply
│       │   └── ...
│       └── test
└── s3dis
    ├── meta_data
    │   ├── area1.txt
    │   ├── area2.txt
    │   └── ...
    └── s3dis_processed
        ├── Area_1
        │   ├── conferenceRoom_1.ply
        │   ├── conferenceRoom_2.ply
        │   └── ...
        ├── Area_2
        └── ...
```

After then, you can train and evalaute a model by using the provided python scripts (`train.py` and `eval.py`) with configuration files in the `config` directory.
For example, you can train and evaluate Fast Point Transformer with voxel size 4cm on S3DIS dataset via the following commands:
```bash
(fpt) ~/FastPointTransformer$ python train.py config/s3dis/train_fpt.gin
(fpt) ~/FastPointTransformer$ python eval.py config/s3dis/eval_fpt.gin {checkpoint_file} # use -r option for rotation averaging.
```

### (Consistency Score) Evaluation
You need to generate predictions via the following command:
```bash
(fpt) ~/FastPointTransformer$ python -m src.cscore.prepare {checkpoint_file} -m {model_name} -v {voxel_size} # This takes hours.
```
Then, you can calculate the consistency score (CScore) with:
```bash
(fpt) ~/FastPointTransformer$ python -m src.cscore.calculate {prediction_dir} # This takes seconds.
```

## Acknowledment

Our code is based on the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).
We also thank [Hengshuang Zhao](https://hszhao.github.io/) for providing [the code](https://github.com/POSTECH-CVLab/point-transformer) of [Point Transformer](https://arxiv.org/abs/2012.09164).
If you use our model, please consider citing them as well.
