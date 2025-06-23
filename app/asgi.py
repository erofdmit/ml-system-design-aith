from main import create_app
import uvicorn
import dotenv 


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", port='127.0.0.1', host=12345, reload=True, log_level="error")