FROM python:3.11-slim

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# Copy project
COPY . .

CMD ["uvicorn", "app.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
