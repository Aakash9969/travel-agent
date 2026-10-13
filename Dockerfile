# Runs anywhere Docker runs: Render, Railway, Fly.io, Hugging Face Spaces,
# a VPS, etc. The port comes from $PORT (set by most platforms; falls back
# to 8000 locally).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
