FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]

ENV AWS_REGION="us-east-1"
ENV USE_GPU=0
ENV DEBIAN_FRONTEND=noninteractive

ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY
ARG WANDB_PROJECT
ENV WANDB_PROJECT=$WANDB_PROJECT
ARG WANDB_USERNAME
ENV WANDB_USERNAME=$WANDB_USERNAME

ENV WANDB_DISABLE_CODE=false

USER root
RUN apt update && apt-get install -y build-essential libglib2.0-0 curl

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN curl -Ok https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh && /bin/bash Anaconda3-2020.11-Linux-x86_64.sh -b -p ~/anaconda
ENV PATH=~/anaconda/bin:$PATH
RUN conda update conda

RUN conda create -n venv python=3.8 && source activate venv && \
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html && \
pip install torch-scatter torch-cluster torch-spline-conv torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cpu.html && \
pip install -r /app/requirements.txt

RUN mkdir models
RUN mkdir output

COPY traffik /app/traffik
COPY setup.py /app/setup.py

WORKDIR /app
RUN pip install .
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
