from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    index_path: str = "index.bin"

    model_config = {
        "env_file": ".env",
        "env_prefix": "FINDEX_"
    }


settings = Settings()