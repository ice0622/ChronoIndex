# ChronoIndex 実行手順書

## 全体フロー

```
YouTube URL
  → [Step 1] 音声ダウンロード（yt-dlp）
  → [Step 2] 文字起こし（Groq Whisper API）  ← レートリミットで複数時間かかる場合あり
  → [Step 3] 境界検出
  → [Step 4] チャプタータイトル生成（LLM）
  → [Step 5] YouTube形式で出力
```

---

## 前提: .env の設定

```env
# 必須
GROQ_API_KEY=gsk_...         # https://console.groq.com でAPIキー取得（無料）
GEMINI_API_KEY=AIza...       # LLM用（チャプタータイトル生成）

# デフォルトのまま使う設定（変更不要）
ASR_MODE=groq
LLM_PROVIDER=gemini
AUDIO_LANGUAGE=ja

# OpenAI に切り替えたい場合（約$1/168分・レートリミットなし）
# ASR_MODE=openai
# OPENAI_API_KEY=sk-...
```

---

## コマンド

### 1. YouTube URLから一発実行（推奨）

```bash
docker compose run --rm dev python -m src.pipeline \
  --url "https://www.youtube.com/watch?v=XXXX"
```

### 2. dry-run（文字起こしまで・LLMコストなし）

```bash
docker compose run --rm dev python -m src.pipeline \
  --url "https://www.youtube.com/watch?v=XXXX" \
  --dry-run
```

### 3. 音声ファイルが手元にある場合

```bash
docker compose run --rm dev python -m src.pipeline \
  --audio "/app/data/audio/<ファイル名>.mp3"
```

### 4. 出力をファイルに保存

```bash
docker compose run --rm dev python -m src.pipeline \
  --url "https://www.youtube.com/watch?v=XXXX" \
  --output /app/data/output/chapters.txt
```

---

## レートリミット（Groq 無料枠）の挙動

- **制限**: 1時間あたり 7,200秒（= 120分）の音声
- **168分の動画**: 2回に分けて処理される（自動で待機→リトライ）
- **ログの見方**:
  ```
  レートリミット到達 → 291秒後に自動リトライ (1/3回目)
  待機中... 残り 291秒
  待機中... 残り 261秒
  ...
  [チャンク 13/17] ... Groq API に送信中...  ← 自動再開
  ```
- キャッシュ済みチャンクはスキップされるので、中断しても次回から続き

---

## トラブルシューティング

### `GROQ_API_KEY が未設定` エラー
→ `.env` に `GROQ_API_KEY=gsk_...` を追加

### リトライ上限(3回)に達してしまう
→ 1時間後に同じコマンドを再実行（キャッシュで続きから）

### 文字起こしのキャッシュをリセットしたい
```bash
rm data/transcripts/transcript_cache.db
```

### ASR を OpenAI に切り替える（レートリミットなし・$0.006/分）
```env
ASR_MODE=openai
OPENAI_API_KEY=sk-...
```

---

## サンプル実行ログ（正常時）

```
INFO  | ChronoIndex 開始 | ASR_MODE=groq | LLM_PROVIDER=gemini
INFO  | [Step 1] 音声ダウンロード: https://youtube.com/watch?v=...
INFO  |   → タイトル (3600秒) 保存先: /app/data/audio/タイトル.mp3
INFO  | [Step 2] 音声認識: /app/data/audio/タイトル.mp3
INFO  |   モード: groq / モデル: small / 言語: ja
INFO  |   総再生時間: 60.0分 / チャンク数: 6 / キャッシュ済み: 0
INFO  |   [チャンク 1/6] 0.0分 〜 10.0分 Groq API に送信中...
INFO  |     → 20セグメント取得・保存完了
  ...
SUCCESS | Groq 文字起こし完了: 800セグメント / 総時間: 60.0分
INFO  | [Step 3] 境界検出
INFO  | [Step 4] チャプタータイトル生成
INFO  | [Step 5] 出力

==================================================
# チャプターリスト（YouTube 概要欄用）
==================================================
0:00  オープニング
5:30  自己紹介・背景説明
  ...
```
