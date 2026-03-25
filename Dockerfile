FROM python:3.11-slim

# --------------------
# システム依存パッケージ
# --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    libsndfile1 \
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
# PyTorch CPU-only インストール（GPU なし環境用）
# GPU 環境に移行する場合は --index-url を削除して nvidia base image に切り替える
# --------------------
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# --------------------
# requirements.txt のインストール
# --------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------
# spaCy 日本語モデルのダウンロード
# --------------------
RUN python -m spacy download ja_core_news_sm

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
