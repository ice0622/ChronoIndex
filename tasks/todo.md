# ChronoIndex タスク管理

## 現在のフェーズ: Phase 3 - チャンクキャッシュ設計

### 完了済み
- [x] `src/transcript_cache.py` 新規作成（SQLite によるチャンクキャッシュ）
- [x] `src/transcribe.py` をキャッシュ対応に全面書き替え
  - チャンクサイズ 20分 → 10分 に変更
  - チャンクごとに DB 保存 → 途中失敗時に再実行で続きから再開
  - `_call_groq_api_raw()` で dict 返却 → JSON で DB 保存

### 次のタスク
- [ ] 動作確認: `docker compose run --rm dev python -m src.pipeline --audio <ファイル> --dry-run`
  - キャッシュなし → 全チャンク処理されること
  - 再実行 → 全チャンクスキップ（0秒で完了）されること

---

## Phase 0 - 環境構築（完了）

- [x] `tasks/todo.md` 作成
- [x] `requirements.txt` 作成
- [x] `Dockerfile` 作成
- [x] `docker-compose.yml` 作成

## Phase 1 - ベースラインパイプライン（完了）

- [x] `src/extract_audio.py`
- [x] `src/transcribe.py`
- [x] `src/detect_boundaries.py`
- [x] `src/generate_chapters.py`
- [x] `src/pipeline.py`

---

## レビューセクション

### 2026-03-30 高速化方針の決定
- ボトルネックは Step 2（Whisper 文字起こし）が全体の 90% 以上
- Groq Whisper API（無料・7,200分/日）で 20〜30倍の高速化が見込める
- 60分動画: ローカルCPU 25〜40分 → Groq API 約1〜2分
- asr_mode=groq をデフォルトに変更済み（ローカルは asr_mode=local で維持）
