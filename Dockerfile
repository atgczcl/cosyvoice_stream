FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tar \
    wget \
    git \
    bash \
    vim

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"

# 接受 Anaconda 服务条款并配置 conda
RUN conda config --add channels conda-forge
# RUN conda config --set anaconda_terms_accepted yes
RUN conda tos accept --override-channels --channel conda-forge

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 安装 Python 3.10
RUN conda install python==3.10

# 复制代码并设置工作目录
COPY . /root/CosyVoice
WORKDIR /root/CosyVoice

# 初始化子模块
RUN git submodule update --init --recursive

# 升级 pip 并安装依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 安装额外依赖
RUN pip install flask waitress

# 设置环境变量
ENV PYTHONPATH=/root/CosyVoice/third_party/Matcha-TTS
ENV API_HOST=0.0.0.0
ENV API_PORT=8080