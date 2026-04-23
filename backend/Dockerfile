FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir \
        fastapi>=0.115.0 \
        "uvicorn[standard]>=0.32.0" \
        chromadb-client>=0.6.0

COPY backend/ ./backend/

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
