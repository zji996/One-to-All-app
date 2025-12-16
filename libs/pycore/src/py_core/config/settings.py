from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = Field(default="dev", alias="ENV")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    models_dir: str = Field(default="models", alias="MODELS_DIR")
    one_to_all_model_dir: str = Field(
        default="One-to-All-14b-FP8", alias="ONE_TO_ALL_MODEL_DIR"
    )
    one_to_all_model_repo_id: str = Field(
        default="MochunniaN1/One-to-All-14b", alias="ONE_TO_ALL_MODEL_REPO_ID"
    )

    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND"
    )
    celery_default_queue: str = Field(default="one_to_all", alias="CELERY_DEFAULT_QUEUE")

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/one_to_all",
        alias="DATABASE_URL",
    )

    s3_endpoint: str | None = Field(default=None, alias="S3_ENDPOINT")
    s3_access_key: str | None = Field(default=None, alias="S3_ACCESS_KEY")
    s3_secret_key: str | None = Field(default=None, alias="S3_SECRET_KEY")
    s3_bucket_name: str | None = Field(default=None, alias="S3_BUCKET_NAME")
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")
    s3_secure: bool = Field(default=False, alias="S3_SECURE")
    s3_prefix: str = Field(default="one-to-all", alias="S3_PREFIX")

    one_to_all_animation_dir: str = Field(
        default="third_party/One-to-All-Animation", alias="ONE_TO_ALL_ANIMATION_DIR"
    )
    one_to_all_animation_runtime_dir: str = Field(
        default="data/one_to_all_animation_runtime",
        validation_alias=AliasChoices(
            "ONE_TO_ALL_ANIMATION_RUNTIME_DIR",
            "ONE_TO_ALL_RUNTIME_DIR",
        ),
    )
    one_to_all_animation_pretrained_dir: str = Field(
        default="models/One-to-All-14b/pretrained_models",
        validation_alias=AliasChoices(
            "ONE_TO_ALL_ANIMATION_PRETRAINED_DIR",
            "ONE_TO_ALL_PRETRAINED_DIR",
        ),
    )
    one_to_all_wan_t2v_14b_diffusers_dir: str = Field(
        default="models/One-to-All-14b/pretrained_models/Wan2.1-T2V-14B-Diffusers",
        validation_alias=AliasChoices(
            "WAN_T2V_14B_DIFFUSERS_DIR",
            "ONE_TO_ALL_WAN_T2V_14B_DIFFUSERS_DIR",
        ),
    )


settings = Settings()
