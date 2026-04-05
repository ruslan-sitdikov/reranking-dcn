FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu124.py310:latest

WORKDIR /app

RUN curl -LsSf https://github.com/astral-sh/uv/releases/download/0.8.22/uv-installer.sh > install.sh && \
    echo "f1ac30b1849f90b17ca93a9c2d40f74d7ad79cf57ad5a179bc5c9948848a70cd install.sh" | sha256sum --check && \
    bash install.sh && \
    rm install.sh

ENV PATH="/app/.venv/bin:/root/.cargo/bin:/root/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"
RUN ldconfig

COPY vertexai-training/reranking_dcn/requirements/production.txt requirements.txt
RUN uv venv
RUN uv pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.5.1" triton --no-cache
RUN uv pip install -r requirements.txt --no-cache

COPY vertexai-training/reranking_dcn/src src
COPY vertexai-training/reranking_dcn/configs configs
COPY vertexai-training/reranking_dcn/sky_configs sky_configs

ENV PYTHONPATH="/app/src:${PYTHONPATH}"
