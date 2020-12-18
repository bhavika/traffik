FROM anibali/pytorch:1.7.0-cuda11.0-ubuntu20.04
SHELL ["/bin/bash", "-c"]

ENV AWS_REGION="us-east-1"
ENV USE_GPU=1
ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt update && apt-get install -y build-essential libglib2.0-0

WORKDIR /app
RUN conda create -n venv python=3.8
RUN source activate venv

COPY requirements.txt /app/requirements.txt

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html -r /app/requirements.txt

RUN mkdir models
RUN mkdir output

COPY src /app/src
COPY setup.py /app/setup.py

RUN pip install .
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8