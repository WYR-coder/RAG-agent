from pathlib import Path
from pydantic_settings import BaseSettings

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8"}

    zhipu_api_key: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    milvus_uri: str = "./data/milvus_lite.db"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8080
    volc_api_key: str = ""
    volc_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    volc_vision_model: str = "doubao-seed-2-0-pro-260215"


settings = Settings()
