# Use official miniforge base image
FROM condaforge/miniforge3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH

# Update system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a conda environment with all required packages
COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean --all -y

# Activate the environment by default
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate torch_env" >> ~/.bashrc

ENV PATH="/opt/conda/envs/torch_env/bin:$PATH"
