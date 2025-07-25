# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base
RUN rm -rf /var/lib/apt/lists/*
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list
RUN apt-get update -o Acquire::http::No-Cache=True -o Acquire::http::Pipeline-Depth=0
RUN apt-get install -y \
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

# Install requirements
RUN conda config --add channels conda-forge
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda install -y python=3.10
# RUN git clone https://gitee.com/atgczcl/cosyvoice_stream.git /root/CosyVoice

COPY . /root/CosyVoice

WORKDIR /root/CosyVoice
RUN git submodule update --init --recursive
RUN apt-get install -y libsndfile1 libsndfile1-dev
RUN pip install -r requirements.txt
# RUN pip install flask waitress flask-cors

# Set environment variables
ENV PYTHONPATH=third_party/Matcha-TTS
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Run
# COPY download_model.py .
RUN python download_model.py
#拷贝文件夹D:\AI\Voice\CosyVoice\pretrained_models
# COPY pretrained_models ./pretrained_models
# COPY index.html .
# COPY cosyvoice.py ./cosyvoice/cli/cosyvoice.py
# COPY app.py .
# CMD ["python", "app.py"]