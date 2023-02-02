# face-extractor

# Face Extractor #
This is a deep learning library that makes face recognition efficient.

****
## Contents
* [Pre-Requisites](#markdown-header-pre-requisites)
* [Requirements](#markdown-header-requirements)
* [Docker](#markdown-header-docker)
* [Usage](#markdown-header-usage)
* [Applications](#markdown-header-applications)
* [Training and Validation](#markdown-header-training-and-validation)
* [Data Zoo](#markdown-header-data-zoo)
* [Model Zoo](#markdown-header-model-zoo)
* [Achievement](#markdown-header-achievement)
* [Acknowledgement](#markdown-header-acknowledgement)
* [Citation](#markdown-header-citation)

****
## Pre-Requisites

* Ubuntu 20.04
* Python 3.6.10

****
## Requirements
```
pip3 install -r requirements.txt
```

****
## Docker
```
bash docker/run.sh
```

****
## Usage

* Clone the repo: `git clone git@github.com:tantm97/face-extractor.git`.
* Prepare your train/val/test data, and ensure database folder and list have the following structure:
  ```
  ./data/db_name/
          -> id1/
              -> 1.jpg
              -> ...
          -> id2/
              -> 1.jpg
              -> ...
          -> ...
              -> ...
              -> ...
  .data_list.txt
          db_name/id1/1.jpg 0
          db_name/id1/2.jpg 0
          db_name/id1/3.jpg 0
          ...
          db_name/id3/1.jpg 3
          db_name/id4/2.jpg 4
  ```
* Config parameters in configs/base.yml.

****
## Applications

****
## Training and Validation
* Testing model.
```
python3 test.py --config <PATH_TO_CONFIG>
Ex: python3 test.py --config configs/base.yml
```

****
## ONNX:

****
## Data Zoo

****
## Model Zoo

****
## Achievement

****
## Acknowledgement

This repo is inspired by:

* [Arcface.Pytorch](https://github.com/ronghuaiyang/arcface-pytorch/tree/47ace80b128042cd8d2efd408f55c5a3e156b032).
* [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe/tree/9a3af4b5eabfd3603ae3a8edfce0775bdb0f78dc).
* [InsightFace.Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch/tree/350ff7aa9c9db8d369d1932e14d2a4d11a3e9553).
* [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo).
* [Arcface.Torch.Mxnet](https://github.com/deepinsight/insightface/tree/8c0aa84c6234065e9e52298ed30821a8f89d0424/recognition/arcface_torch).
* [Cavaface](https://github.com/cavalleria/cavaface).
* [CurricularFace](https://github.com/HuangYG123/CurricularFace).
* [PretrainedModels.PyTorch](https://github.com/Cadene/pretrained-models.pytorch).
* [ArcFace-MXNet-Gluon](https://github.com/haleuh/ArcFace-MXNet-Gluon).
* [Angular Penalty Softmax Losses Pytorch](https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/tree/c41d599622b6a6ba7e5be6faf3a01da1202024ea).
* [facenet-pytorch-glint360k](https://github.com/tamerthamoqa/facenet-pytorch-glint360k).
* [Mag Face](https://github.com/IrvingMeng/MagFace).

Scheduler:

* [Learning Rate Scheduling](https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling).

Loss:

* [Focal loss](https://phamdinhkhanh.github.io/2020/08/23/FocalLoss.html).

Others:

* [Effect of batch size on training dynamics](https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e).
* [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#regularfaceregularizer).
* [ONEFLOW](https://chowdera.com/2021/03/20210303193428334W.html).
* [Device side assert triggered](https://programmerah.com/solved-runtimeerror-cuda-error-device-side-assert-triggered-30474/).
* [Resize the usable area of a tmux session](https://superuser.com/questions/880497/how-do-i-resize-the-usable-area-of-a-tmux-session).
* [Interpreting negative cosine similarity](https://stats.stackexchange.com/questions/198810/interpreting-negative-cosine-similarity).
* [Arccos](https://www.mathopenref.com/arccos.html).

Coding style guide:

* [Google style guide](https://google.github.io/styleguide/pyguide.html).
* [Summary](https://gist.github.com/lneeraj97/8f617b1f67434b11a9f491f8b202eda9).

****
## Citation
