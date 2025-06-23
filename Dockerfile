FROM python:3.11.9-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

COPY ./custom_inference ./custom_inference
COPY ./static ./static
COPY main.py asgi.py streamlit_app.py ./

CMD ["uvicorn", "asgi:app", "--host", "0.0.0.0", "--port", "8000"]
