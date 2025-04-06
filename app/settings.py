from pathlib import Path

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent

    class Config:
        env_file: str = '.env'
        env_file_encoding: str = 'utf-8'


class DBConfig(AppSettings):
    HOST: str
    PORT: str
    USER: str
    PASS: str
    NAME: str
    ECHO: bool

    class Config:
        env_prefix: str = 'DB_'

    def generate_db_url(self):
        return f'postgresql+asyncpg://{self.USER}:{self.PASS}@{self.HOST}:{self.PORT}/{self.NAME}'


class Settings(AppSettings):
    ROOT_PATH: str = '/api'
    DB: DBConfig = DBConfig()
    DB_URL: str = DB.generate_db_url()

    class Config:
        env_file: str = '.env'
        env_file_encoding: str = 'utf-8'


settings = Settings()
