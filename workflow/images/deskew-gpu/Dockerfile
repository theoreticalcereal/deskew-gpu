FROM docker.io/continuumio/miniconda3:24.11.1-0

COPY environment.yml /tmp/environment.yml

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libice6 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxmu6 \
        libxrender1 \
        libxt6 \
        procps \
    && rm -rf /var/lib/apt/lists/* \
    && conda env create --name app --file /tmp/environment.yml \
    && conda clean --all --yes

ENV CONDA_PREFIX=/opt/conda/envs/app
ENV CONDA_DEFAULT_ENV=app
ENV PATH=/opt/conda/envs/app/bin:/opt/conda/bin:$PATH
ENV PYTHONNOUSERSITE=1

WORKDIR /workspace
ENTRYPOINT ["python"]
