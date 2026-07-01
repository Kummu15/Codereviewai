from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "CodeReview AI"
    PROJECT_VERSION: str = "0.1.0"
    API_PREFIX: str = ""


settings = Settings()
