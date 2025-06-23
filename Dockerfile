FROM python:3.11.9-slim

# Install system dependencies for OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# Copy project
COPY ./custom_inference ./custom_inference
COPY ./static ./static
COPY main.py asgi.py streamlit_app.py ./

CMD ["uvicorn", "asgi:app", "--host", "0.0.0.0", "--port", "8000"]
