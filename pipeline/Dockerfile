FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements-dev.txt /app/requirements-dev.txt
RUN pip install --upgrade pip && pip install -r /app/requirements-dev.txt

COPY config /app/config
COPY src /app/src
COPY scripts /app/scripts
COPY tests /app/tests
COPY data/raw /app/data/raw

CMD ["bash", "scripts/run_pipeline.sh"]
