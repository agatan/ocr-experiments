FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get -qq update -y \
 && apt-get -qq install -y software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get -qq update -y \
 && echo "Install python3.6 and some libraries..." \
 && apt-get -qq install -y wget bzip2 python3.6 python3.6-dev build-essential libsm6 libxrender1 libxext6 git cmake \
 && echo "Install pip..." \
 && cd /tmp \
 && wget https://bootstrap.pypa.io/get-pip.py \
 && python3.6 get-pip.py \
 && rm /tmp/* \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

