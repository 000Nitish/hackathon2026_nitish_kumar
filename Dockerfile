FROM python:3.11-slim

RUN groupadd -r shopwave && useradd -r -g shopwave shopwave

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY data/ ./data/
COPY main.py .

RUN mkdir -p /app/output && chown -R shopwave:shopwave /app

USER shopwave

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import os; assert os.getenv('GOOGLE_API_KEY'), 'API key missing'; from src.tools import DataStore; DataStore.get()"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
