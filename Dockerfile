# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base
# FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

# # 安装基本工具
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     tar \
#     wget \
#     git \
#     bash \
#     vim

# # 安装 Miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir -p /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh
# ENV PATH="/root/miniconda3/bin:${PATH}"

# # 接受 conda-forge 频道的 Terms of Service
# # RUN conda tos accept --override-channels --channel conda-forge

# # 创建并激活 Cosyvoice_stream 环境
# RUN conda create -n Cosyvoice_stream python=3.10 -y
# ENV PATH="/root/miniconda3/envs/Cosyvoice_stream/bin:${PATH}"


FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing
RUN apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev && \
    apt-get clean && \
    git lfs install

# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH /opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
# ------------------------------------------------------------------
# ~conda
# ==================================================================

RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH /opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace

ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

# RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
COPY . /workspace/CosyVoice

RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5
RUN conda activate ${VENV} && cd CosyVoice && \
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

WORKDIR /workspace/CosyVoice
ENV API_HOST=0.0.0.0
ENV API_PORT=8080




# 安装项目依赖
# COPY . /root/CosyVoice
# WORKDIR /root/CosyVoice
# RUN conda install --file requirements.txt

# # 更新子模块
# RUN git submodule update --init --recursive

# # 安装其他 Python 依赖
# RUN pip install -r requirements.txt

# # 设置环境变量
# ENV PYTHONPATH=third_party/Matcha-TTS
# ENV API_HOST=0.0.0.0
# ENV API_PORT=8080

# # 运行项目
# RUN python download_model.py
# COPY pretrained_models ./pretrained_models
# COPY index.html .
# COPY cosyvoice.py ./cosyvoice/cli/cosyvoice.py
# COPY app.py .
# CMD ["python", "app.py"]
