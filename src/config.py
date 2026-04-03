"""
設定管理モジュール
.env ファイルの環境変数を Pydantic モデルとして管理する
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API キー
    openai_api_key:     str = Field(default="")
    anthropic_api_key:  str = Field(default="")
    huggingface_token:  str = Field(default="")
    groq_api_key:       str = Field(default="")   # Groq Whisper API（無料・高速）

    # ASR 設定
    # asr_mode: "groq"(推奨・無料・高速) | "api"(OpenAI有料) | "local"(ローカルCPU)
    asr_mode:           str = Field(default="groq")
    whisper_model_size: str = Field(default="small")
    # Groq Whisper モデル選択
    # whisper-large-v3-turbo: 最速・精度十分（推奨）
    # whisper-large-v3: 最高精度・やや遅い
    groq_whisper_model: str = Field(default="whisper-large-v3-turbo")

    # LLM 設定
    llm_provider:       str   = Field(default="gemini")
    openai_model:       str   = Field(default="gpt-4o-mini")
    anthropic_model:    str   = Field(default="claude-3-5-haiku-20241022")
    gemini_api_key:     str   = Field(default="")
    gemini_model:       str   = Field(default="gemini-2.5-flash")
    llm_temperature:    float = Field(default=0.1)

    # 処理設定
    audio_language:         str   = Field(default="ja")
    window_size_sec:        int   = Field(default=3600)
    blank_min_silence_sec:  int   = Field(default=30)
    data_dir:               Path  = Field(default=Path("/app/data"))

    # 話者ダイアライゼーション設定
    # diarization_enabled=true にすると pyannote.audio で話者分離を実行する
    # 前提: HUGGINGFACE_TOKEN が設定済み + requirements-local.txt インストール済み
    diarization_enabled:    bool       = Field(default=False)
    # num_speakers を指定すると精度が上がる（None = 自動推定）
    diarize_num_speakers:   int | None = Field(default=None)


def load_config() -> Config:
    """設定を読み込んで返す"""
    return Config()
