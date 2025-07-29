FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# 设置阿里云镜像源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

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

ENV PATH=/opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
# ------------------------------------------------------------------
# ~conda
# ==================================================================
# 在现有 conda 配置后添加
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set channel_priority strict && \
    conda config --set ssl_verify false && \
    conda config --set remote_connect_timeout_secs 60.0 && \
    conda config --set remote_read_timeout_secs 120.0


RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace

RUN export PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

# RUN git clone --recursive https://github.com/journey-ad/CosyVoice.git
COPY . /workspace/CosyVoice

RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5
RUN conda activate ${VENV} && cd CosyVoice && \
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

WORKDIR /workspace/CosyVoice

ENV LOG_LEVEL=INFO

EXPOSE 8080

# CMD ["sh", "-c", "python webui.py --port 8080 --log_level ${LOG_LEVEL}"]



ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Run
# COPY download_model.py .
# RUN python download_model.py
#拷贝文件夹D:\AI\Voice\CosyVoice\pretrained_models
# COPY pretrained_models ./pretrained_models
# COPY index.html .
# COPY cosyvoice.py ./cosyvoice/cli/cosyvoice.py
# COPY app.py .
# CMD ["python", "app.py"]
CMD ["python", "Empty.py"]