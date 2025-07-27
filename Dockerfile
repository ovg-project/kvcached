FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV ENABLE_KVCACHED=true
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y \
        software-properties-common git curl wget build-essential ninja-build \
        python3.11 python3.11-dev python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*       \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip setuptools wheel

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
RUN git clone https://github.com/ovg-project/kvcached.git
WORKDIR /workspace/kvcached
RUN pip install . --no-build-isolation

WORKDIR /workspace/kvcached/engine_integration/scripts
RUN chmod +x setup.sh && ./setup.sh all

EXPOSE 8080 30000 30001

WORKDIR /workspace/kvcached
CMD ["bash"]
