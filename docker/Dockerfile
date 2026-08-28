FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://dl.openfoam.org/gpg.key \
    > /etc/apt/trusted.gpg.d/openfoam.asc \
    && add-apt-repository "http://dl.openfoam.org/ubuntu main dev" \
    && apt-get update \
    && apt-get install -y --no-install-recommends openfoam14 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ARCHITECTURE.md V2_CHANGES.md /workspace/
COPY src /workspace/src
COPY examples /workspace/examples
COPY config /workspace/config
COPY tests /workspace/tests
COPY research /workspace/research

RUN python3 -m pip install --break-system-packages -e ".[dev]"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
