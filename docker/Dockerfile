# Adapted from https://github.com/facebookresearch/detectron2/blob/main/docker/Dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update  && \
    apt-get install -y python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# JAVA for pycocoevalcap
RUN sudo apt-get install default-jre -y
RUN sudo apt-get install default-jdk -y

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# setup pip
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py --user && \
    rm get-pip.py

# install a few dependencies
RUN pip install --user torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --user cython pycocotools pycocoevalcap
RUN pip install --user tqdm termcolor pandas h5py faiss-gpu
RUN pip install --user -U 'git+https://github.com/facebookresearch/iopath'

# install LAVIS v1.0.2
RUN pip install --user salesforce-lavis==1.0.2
