"""ASGI entrypoint for the FastAPI application."""

from app.main import create_app
import uvicorn

app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=12345,
        reload=True,
        log_level="error",
    )
