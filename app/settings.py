# app/settings.py — Pydantic v2 compatible
import os
from functools import lru_cache
from typing import List, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ===== API =====
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    chat_list_max: int = Field(default=3000, alias="CHAT_LIST_MAX")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:4200", "http://localhost:5173"],
        alias="CORS_ORIGINS"
    )

    enable_admin_routes: bool = Field(default=False, alias="ENABLE_ADMIN_ROUTES")

    # ===== Milvus =====
    milvus_host: str = Field(default="127.0.0.1", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection: str = Field(default="products_latam", alias="MILVUS_COLLECTION")
    milvus_dim: int = Field(default=768, alias="MILVUS_DIM")

    # ¡IMPORTANTE! que coincida con tu colección
    vector_field: str = Field(default="vector", alias="VECTOR_FIELD")

    metric_type: str = Field(default="IP", alias="METRIC_TYPE")  # índice HNSW en IP
    nprobe: int = Field(default=16, alias="NPROBE")

    # ===== Modelos =====
    ollama_host: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_HOST")

    # Embeddings: "hf" o "ollama"
    embed_backend: Literal["hf","ollama"] = Field(default="hf", alias="EMBED_BACKEND")
    embed_model: str = Field(default="intfloat/multilingual-e5-base", alias="EMBED_MODEL")

    # Generador (lo que vamos a cambiar en los tests)
    gen_model: str = Field(default="phi3:mini", alias="GEN_MODEL")
    gen_temperature: float = Field(default=0.35, alias="GEN_TEMPERATURE")
    gen_num_ctx: int = Field(default=1024, alias="GEN_NUM_CTX")
    gen_num_predict: int = Field(default=256, alias="GEN_NUM_PREDICT")

    compare_llm: bool = Field(default=True, alias="COMPARE_LLM")

    # Umbral / TopK
    abstain_threshold: float = Field(default=0.35, alias="ABSTAIN_THRESHOLD")
    abstention_threshold: float = Field(default=0.35, alias="ABSTENTION_THRESHOLD")
    top_k: int = Field(default=5, alias="TOP_K")

    no_data_message: str = Field(default="No tengo esa información en la base", alias="NO_DATA_MESSAGE")

    # Híbrido
    bm25_enabled: bool = Field(default=False, alias="BM25_ENABLED")

    # Campos
    pk_field: str = Field(default="product_id", alias="PK_FIELD")
    name_field: str = Field(default="name", alias="NAME_FIELD")
    brand_field: str = Field(default="brand", alias="BRAND_FIELD")
    category_field: str = Field(default="category", alias="CATEGORY_FIELD")
    store_field: str = Field(default="store", alias="STORE_FIELD")
    country_field: str = Field(default="country", alias="COUNTRY_FIELD")
    price_field: str = Field(default="price", alias="PRICE_FIELD")
    currency_field: str = Field(default="currency", alias="CURRENCY_FIELD")
    size_field: str = Field(default="size", alias="SIZE_FIELD")
    unit_field: str = Field(default="unit", alias="UNIT_FIELD")
    url_field: str = Field(default="url", alias="URL_FIELD")
    canonical_text_field: str = Field(default="canonical_text", alias="CANONICAL_TEXT_FIELD")

    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("metric_type", mode="after")
    @classmethod
    def upper_metric(cls, v: str):
        return (v or "").upper()

@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # Mantén ambas variantes sincronizadas por compatibilidad con tu .env
    if "ABSTAIN_THRESHOLD" in os.environ:
        s.abstention_threshold = s.abstain_threshold
    elif "ABSTENTION_THRESHOLD" in os.environ:
        s.abstain_threshold = s.abstention_threshold

    # ---- Log útil al arrancar (para tus benchmarks) ----
    print(
        "[SETTINGS]"
        f" GEN_MODEL={s.gen_model}"
        f" | GEN_TEMPERATURE={s.gen_temperature}"
        f" | GEN_NUM_CTX={s.gen_num_ctx}"
        f" | GEN_NUM_PREDICT={s.gen_num_predict}"
        f" | OLLAMA_HOST={s.ollama_host}"
        f" | EMBED_BACKEND={s.embed_backend}"
        f" | EMBED_MODEL={s.embed_model}"
        f" | TOP_K={s.top_k}"
        f" | ABSTAIN={s.abstain_threshold:.2f}"
        f" | BM25_ENABLED={s.bm25_enabled}"
    )
    return s
