FROM python:3.11-slim

WORKDIR /app

# system deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 && rm -rf /var/lib/apt/lists/*

# install poetry
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

COPY . .

CMD ["uvicorn", "asgi:app", "--host", "0.0.0.0", "--port", "8000"]
