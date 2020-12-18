FROM anibali/pytorch:latest
SHELL ["/bin/bash", "-c"]

ENV AWS_REGION="us-east-1"
ENV USE_GPU=1
ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt update && apt-get install -y build-essential libglib2.0-0

WORKDIR /app
RUN conda create -n venv python=3.8
RUN source activate venv
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

RUN TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
RUN CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
ENV TORCH_VERSION=$TORCH_VERSION
ENV CUDA_VERSION=$CUDA_VERSION

COPY src /app/src
COPY setup.py /app/setup.py
COPY requirements.txt /app/requirements.txt

RUN mkdir models
RUN mkdir output

RUN pip install -r /app/requirements.txt
RUN pip install .
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8