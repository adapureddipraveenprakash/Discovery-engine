FROM python:3.11-slim

WORKDIR /app

# System deps for FAISS / numerical libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Default command serves the API. Override for training:
#   docker run <image> python train.py
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
