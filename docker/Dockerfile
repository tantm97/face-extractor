FROM nvidia/cuda:11.5.0-cudnn8-devel-ubuntu20.04
USER root
WORKDIR /workspace
COPY install.sh /workspace
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install python3.8 python3.8-dev python3.8-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN /workspace/install.sh
