# latest pytorch image with cuda
FROM nvcr.io/nvidia/pytorch:24.12-py3
ARG DEBIAN_FRONTEND=noninteractive

# please write your cuda arch
ENV TORCH_CUDA_ARCH_LIST="8.6"

# install dependencies
RUN pip install --upgrade pip
RUN apt update && apt upgrade -y vim git unzip wget
RUN pip install einops
RUN pip install timm
RUN pip install basicsr==1.3.4.9
RUN pip install git+https://github.com/HPC-Lab-KOREATECH/FGA-SR