# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

# 安装基本工具
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tar \
    wget \
    git \
    bash \
    vim

# 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir -p /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"

# 接受 conda-forge 频道的 Terms of Service
# RUN conda tos accept --override-channels --channel conda-forge

# 创建并激活 Cosyvoice_stream 环境
RUN conda create -n Cosyvoice_stream python=3.10 -y
ENV PATH="/root/miniconda3/envs/Cosyvoice_stream/bin:${PATH}"

# 安装项目依赖
COPY . /root/CosyVoice
WORKDIR /root/CosyVoice
RUN conda install --file requirements.txt

# 更新子模块
RUN git submodule update --init --recursive

# 安装其他 Python 依赖
RUN pip install -r requirements.txt

# 设置环境变量
ENV PYTHONPATH=third_party/Matcha-TTS
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# 运行项目
RUN python download_model.py
# COPY pretrained_models ./pretrained_models
# COPY index.html .
# COPY cosyvoice.py ./cosyvoice/cli/cosyvoice.py
# COPY app.py .
# CMD ["python", "app.py"]
