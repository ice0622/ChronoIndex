"""
ChronoIndex Pipeline
YouTube長尺動画をタイムスタンプ付きチャプターリストに変換するエンドツーエンドパイプライン

使用例:
    # インタラクティブ実行
    python -m src.pipeline --url "https://www.youtube.com/watch?v=XXXX"

    # 音声ファイルから直接実行
    python -m src.pipeline --audio /app/data/audio/sample.mp3

    # 出力ファイルを指定
    python -m src.pipeline --url "https://..." --output /app/data/output/chapters.txt
"""

import click
from pathlib import Path
from loguru import logger

from src.config import load_config


@click.command()
@click.option("--url",    "-u", default=None, help="YouTube の動画 URL")
@click.option("--audio",  "-a", default=None, type=click.Path(exists=False), help="既存の音声ファイルパス（yt-dlp をスキップ）")
@click.option("--output", "-o", default=None, type=click.Path(), help="出力テキストファイルのパス（省略時: stdout）")
@click.option("--dry-run", is_flag=True, default=False, help="文字起こしまでを実行して LLM 処理をスキップ（コスト確認用）")
def main(url: str | None, audio: str | None, output: str | None, dry_run: bool) -> None:
    """ChronoIndex: 長尺動画の自動チャプター生成パイプライン"""

    if url is None and audio is None:
        raise click.UsageError("--url または --audio のどちらかを指定してください")

    cfg = load_config()
    logger.info(f"ChronoIndex 開始 | ASR_MODE={cfg.asr_mode} | LLM_PROVIDER={cfg.llm_provider}")

    # -------------------------------------------------------
    # Step 1: 音声ダウンロード（URL 指定時のみ）
    # -------------------------------------------------------
    if url:
        logger.info(f"[Step 1] 音声ダウンロード: {url}")
        # TODO: Phase 1 で実装
        # from src.extract_audio import extract_audio
        # audio = extract_audio(url, output_dir=cfg.data_dir / "audio")
        logger.warning("extract_audio は未実装です（Phase 1 で実装予定）")
        return

    # -------------------------------------------------------
    # Step 2: 音声認識（ASR）
    # -------------------------------------------------------
    logger.info(f"[Step 2] 音声認識: {audio}")
    # TODO: Phase 1 で実装
    # from src.transcribe import transcribe
    # transcript = transcribe(audio, cfg)

    # -------------------------------------------------------
    # Step 3: セグメント境界検出 + 種別分類
    # -------------------------------------------------------
    # TODO: Phase 1 で実装

    # -------------------------------------------------------
    # Step 4: LLM によるチャプタータイトル生成
    # -------------------------------------------------------
    # TODO: Phase 1 で実装

    # -------------------------------------------------------
    # Step 5: YouTube 形式への整形・バリデーション・出力
    # -------------------------------------------------------
    # TODO: Phase 1 で実装

    logger.info("パイプライン完了（スケルトン: 実際の処理は Phase 1 で実装予定）")


if __name__ == "__main__":
    main()
