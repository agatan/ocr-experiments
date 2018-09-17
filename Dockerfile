FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get -qq update \
 && apt-get -qq install python3.7-dev python3-venv python3.7-venv \
                        git cmake \
 && python3.7 -m venv /root/venv

ENV PATH=/root/venv/bin/:$PATH

COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp/

RUN pip install -r requirements.txt \
 && git clone --recursive https://github.com/pytorch/pytorch \
 && cd pytorch \
 && python setup.py install \
 && pip install torchvision
