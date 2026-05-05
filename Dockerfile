FROM python:3.13-slim

LABEL author="Trevor Norton"
LABEL description="Reproducible Python 3.13 container for the classification project."

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/train.py"]
