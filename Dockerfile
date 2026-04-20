# Use multiphase builds for lean images
FROM python:3.11.9-slim AS builder

WORKDIR /app

# Variaveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Instalando requirements de sistema (build de lib C)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar libs do Py
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando Padrão
CMD ["echo", "Aponte o commmando via Docker Compose"]
