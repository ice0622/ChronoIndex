"""
設定管理モジュール
.env ファイルの環境変数を Pydantic モデルとして管理する
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # API キー
    openai_api_key:     str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key:  str = Field(default="", env="ANTHROPIC_API_KEY")
    huggingface_token:  str = Field(default="", env="HUGGINGFACE_TOKEN")
    groq_api_key:       str = Field(default="", env="GROQ_API_KEY")  # Groq Whisper API（無料・高速）

    # ASR 設定
    # asr_mode: "groq"(推奨・無料・高速) | "api"(OpenAI有料) | "local"(ローカルCPU)
    asr_mode:           str = Field(default="groq",  env="ASR_MODE")
    whisper_model_size: str = Field(default="small", env="WHISPER_MODEL_SIZE")
    # Groq Whisper モデル選択
    # whisper-large-v3-turbo: 最速・精度十分（推奨）
    # whisper-large-v3: 最高精度・やや遅い
    groq_whisper_model: str = Field(default="whisper-large-v3-turbo", env="GROQ_WHISPER_MODEL")

    # LLM 設定
    llm_provider:       str   = Field(default="gemini",         env="LLM_PROVIDER")
    openai_model:       str   = Field(default="gpt-4o-mini",    env="OPENAI_MODEL")
    anthropic_model:    str   = Field(default="claude-3-5-haiku-20241022", env="ANTHROPIC_MODEL")
    gemini_api_key:     str   = Field(default="",               env="GEMINI_API_KEY")
    gemini_model:       str   = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    llm_temperature:    float = Field(default=0.1,              env="LLM_TEMPERATURE")

    # 処理設定
    audio_language:         str   = Field(default="ja",    env="AUDIO_LANGUAGE")
    window_size_sec:        int   = Field(default=3600,    env="WINDOW_SIZE_SEC")
    blank_min_silence_sec:  int   = Field(default=30,      env="BLANK_MIN_SILENCE_SEC")
    data_dir:               Path  = Field(default=Path("/app/data"), env="DATA_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def load_config() -> Config:
    """設定を読み込んで返す"""
    return Config()
