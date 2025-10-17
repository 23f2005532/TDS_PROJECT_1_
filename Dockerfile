FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y git gcc libffi-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . /app

ENV PORT=8000
CMD ["sh", "-c", "uvicorn app_new:app --host 0.0.0.0 --port ${PORT:-8000}"]
