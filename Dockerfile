FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url ${TORCH_INDEX_URL} torch torchvision \
    && python -m pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/data /app/checkpoints /app/reports

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "7860", "--server.headless", "true"]
