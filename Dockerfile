# The demo as a web service: the page at /, and /ask, /health and /metrics as JSON.
#
#   docker build -t sicherheitslotse .
#   docker run -p 7860:7860 -e MISTRAL_API_KEY -e HF_TOKEN -v lotse-models:/data sicherheitslotse
#
# At its first start the container downloads both models (about 5 GB) into /data and the index
# from the private dataset, which is what HF_TOKEN is for.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface

WORKDIR /srv
COPY requirements.txt .
# The image has no GPU. The CPU build of torch is taken first, and the CUDA libraries that
# requirements.txt lists for Linux are left out: together they would add several GB.
RUN grep -v -E "^(nvidia-|cuda-|triton)" requirements.txt > cpu.txt \
    && pip install "$(grep '^torch==' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r cpu.txt

COPY app app
COPY src src
COPY data/manifest.json data/manifest.json

# Not root: the app only needs to write the index and the model cache.
RUN useradd --create-home lotse && mkdir -p /data /srv/index && chown -R lotse /data /srv
USER lotse

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "server:api", "--app-dir", "app", "--host", "0.0.0.0", "--port", "7860"]
