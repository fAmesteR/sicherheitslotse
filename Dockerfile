# The demo as a web service: the page at /, and /ask, /health and /metrics as JSON.
#
#   docker build -t sicherheitslotse .
#   docker run -p 7860:7860 -e MISTRAL_API_KEY -e HF_TOKEN -v lotse-models:/data sicherheitslotse
#
# At its first start the container downloads both models (about 5 GB) into /data and the index
# from the private dataset, which is what HF_TOKEN is for. Behind a reverse proxy that sets
# X-Forwarded-For, add -e LOTSE_TRUST_PROXY=1, and let only the proxy reach the port.

# Both images are pinned by digest, so a rebuild starts from the same bytes.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    UV_PROJECT_ENVIRONMENT=/srv/.venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/srv/.venv/bin:$PATH

WORKDIR /srv
COPY pyproject.toml uv.lock ./
# uv installs from uv.lock and checks every package against the hash recorded there. torch and its
# CUDA libraries are left out, which would add several GB to an image without a GPU. torch comes
# instead as the CPU build of the version pyproject.toml pins, checked against PyTorch's hash.
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    uv sync --frozen --no-dev --no-install-project \
        $(grep -oE '^name = "(torch|triton|nvidia-[a-z0-9-]+)"' uv.lock | cut -d'"' -f2 \
          | sed 's/^/--no-install-package /') \
    && echo "torch==2.13.0+cpu --hash=sha256:4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2" > /tmp/torch.txt \
    && uv pip install --python /srv/.venv/bin/python --require-hashes --no-deps -r /tmp/torch.txt \
        --index-url https://download.pytorch.org/whl/cpu \
    && rm /tmp/torch.txt

COPY app app
COPY src src
COPY data/manifest.json data/manifest.json

# Not root: the app only needs to write the index, the model cache and Gradio's own folder in
# /srv. The installed packages stay root's, and are not copied into a second layer by a chown.
RUN useradd --create-home lotse && mkdir -p /data /srv/index \
    && chown lotse /srv && chown -R lotse /data /srv/index
USER lotse

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
# No access log: it would write every visitor's address, and the query log keeps none.
CMD ["uvicorn", "server:api", "--app-dir", "app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
