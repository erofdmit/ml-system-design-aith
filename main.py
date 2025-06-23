from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="Video Inference API")

    @app.get("/ping")
    async def ping():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
