from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    database_url: str = "sqlite:///./food_assistant.db"

    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "foods"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    log_level: str = "INFO"
    max_tool_iterations: int = 5
    llm_timeout_seconds: int = 30


settings = Settings()
