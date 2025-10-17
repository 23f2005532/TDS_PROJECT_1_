# -----------------------
# Base Image
# -----------------------
FROM python:3.12-slim

# -----------------------
# Metadata
# -----------------------
LABEL maintainer="Your Name <you@example.com>"
LABEL description="AI Web App Builder (FastAPI) for Hugging Face Spaces"

# -----------------------
# Environment Defaults
# -----------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    LOG_FILE_PATH="logs/app.log" \
    GITHUB_API_BASE="https://api.github.com"

# -----------------------
# Workdir
# -----------------------
WORKDIR /app

# -----------------------
# Copy App Files
# -----------------------
COPY . /app

# -----------------------
# Install System Dependencies
# -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

# -----------------------
# Install Python Dependencies
# -----------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------
# Create Log Directory
# -----------------------
RUN mkdir -p $(dirname $LOG_FILE_PATH)

# -----------------------
# Expose Port
# -----------------------
EXPOSE ${PORT}

# -----------------------
# Command to Run the App
# -----------------------
CMD ["uvicorn", "app_new:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
