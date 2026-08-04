FROM pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

ARG CONTRACT_SHA256
LABEL org.opencontainers.image.title="aorta-surrogate AWS trainer" \
      org.opencontainers.image.version="aws-preop-aaa-v1" \
      ai.toralis.contract.sha256="${CONTRACT_SHA256}" \
      ai.toralis.runtime-tree.sha256="43000bd90a93afad3fefc41b5cef8cd9b3042a341ea59485138ee44fc6b8f17a"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/aorta

COPY infra/watcloud/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.lock

COPY pyproject.toml ./
COPY aorta_surrogate ./aorta_surrogate
COPY configs/aws_preop_v1_frozen.json ./configs/aws_preop_v1_frozen.json
COPY configs/watcloud_preop_v1_frozen.json ./configs/watcloud_preop_v1_frozen.json
COPY scripts ./scripts
COPY tests ./tests
RUN python -m pip install --no-cache-dir --no-deps . && \
    python -m pip freeze | LC_ALL=C sort > /opt/aorta/environment.lock

CMD ["python", "-m", "aorta_surrogate.training.watcloud_trainer", "--help"]
