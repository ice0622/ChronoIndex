FROM python:3.11-slim

# --------------------
# システム依存パッケージ
# ffmpeg: 音声変換に必須 / curl: ヘルスチェック用
# build-essential・libsndfile1 は ASR_MODE=groq では不要なので除外
# --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------
# 作業ディレクトリ
# --------------------
WORKDIR /app

# --------------------
# Python: pip アップグレード
# --------------------
RUN pip install --upgrade pip

# --------------------
# NOTE: PyTorch は ASR_MODE=groq では不要なため除外。
# ローカル Whisper（ASR_MODE=local）に戻す場合は
# requirements-local.txt を pip install すること。
# --------------------

# --------------------
# requirements.txt のインストール
# --------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------
# ソースコードのコピー
# --------------------
COPY src/ ./src/
COPY evaluate/ ./evaluate/

# --------------------
# 出力ディレクトリの作成
# --------------------
RUN mkdir -p /app/data/audio /app/data/transcripts /app/data/output

# --------------------
# 非 root ユーザーで実行（セキュリティ）
# --------------------
RUN useradd -m -u 1000 chronouser
RUN chown -R chronouser:chronouser /app
USER chronouser

# --------------------
# デフォルトコマンド
# --------------------
CMD ["python", "-m", "src.pipeline", "--help"]
