# ChronoIndex タスク管理

## 現在のフェーズ: Phase 0 - 環境構築

### 環境構築チェックリスト

- [x] `tasks/todo.md` 作成
- [ ] `requirements.txt` 作成
- [ ] `Dockerfile` 作成
- [ ] `docker-compose.yml` 作成
- [ ] `.env.example` 作成
- [ ] `src/` ディレクトリ構成作成
- [ ] `docker build` が通ることを確認
- [ ] `docker compose up` でコンテナ起動確認
- [ ] コンテナ内で `yt-dlp --version` 確認
- [ ] コンテナ内で `python -c "import whisper"` 確認
- [ ] `.env` に実際のAPIキーを設定

### 次のフェーズ: Phase 1 - ベースラインパイプライン実装

- [ ] `src/extract_audio.py` - YouTube URL → 音声ファイル
- [ ] `src/transcribe.py` - 音声 → タイムスタンプ付きトランスクリプト
- [ ] `src/detect_boundaries.py` - 境界検出ロジック
- [ ] `src/generate_chapters.py` - LLMによるタイトル生成
- [ ] `src/format_output.py` - YouTube形式の出力・バリデーション
- [ ] `src/pipeline.py` - エンドツーエンド実行スクリプト
- [ ] 評価用テスト動画3本の収集
- [ ] `evaluate/evaluate_boundary.py` 実装
- [ ] `evaluate/evaluate_asr.py` 実装
- [ ] `evaluate/evaluate_blank.py` 実装

---

## レビューセクション（未記入）

完了後に結果・気づき・改善点をここに記録する。
