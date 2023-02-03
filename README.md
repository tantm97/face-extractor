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
* Download example (weights and samples):
```
bash download.sh
```
OR
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
* Start docker env and run test.py:
```
# Replace the source to project path in your machine.
sudo docker run -itd --name tandev --gpus all  --shm-size=2gb -p 9001:9001 --mount type=bind,source=/home/tan/workspace/git/face-extractor/,target=/workspace aidev-cuda11.5-cudnn8-devel-ubuntu20.04:1.0.0
# Replace tandev to your container name if you changed in the previous command.
docker exec -it tandev bash
python3 test.py
```

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
## Result:
```
Val: 100%|██████████████████████████████████████| 14/14 [00:01<00:00, 10.35it/s]
Threshold: 100%|████████████████████████████| 9/9 [00:00<00:00, 251.47it/s, 0.8]
Accuracy =  0.9821428571428571
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
