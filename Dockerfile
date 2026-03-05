FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

RUN pip install --no-cache-dir torch-geometric==2.4.0

COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

COPY app/ ./app/
COPY data/ ./data/
COPY static/ ./static/
COPY checkpoints/ ./checkpoints/
COPY tests/ ./tests/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
