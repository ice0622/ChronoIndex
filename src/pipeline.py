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
        from src.extract_audio import extract_audio
        audio_info = extract_audio(url, output_dir=cfg.data_dir / "audio")
        audio = str(audio_info.path)
        logger.info(f"  → {audio_info.title} ({audio_info.duration_sec}秒) 保存先: {audio}")

    # -------------------------------------------------------
    # Step 2: 音声認識（ASR）
    # -------------------------------------------------------
    logger.info(f"[Step 2] 音声認識: {audio}")
    # TODO: Phase 1 Step 2 で実装

    # -------------------------------------------------------
    # Step 2: 音声認識（ASR）
    # -------------------------------------------------------
    logger.info(f"[Step 2] 音声認識: {audio}")
    from src.transcribe import transcribe
    from pathlib import Path
    transcript = transcribe(Path(audio), cfg)
    logger.info(f"  → {len(transcript.segments)}セグメント / 総時間: {transcript.duration_sec/60:.1f}分")

    if dry_run:
        logger.info("[dry-run] ここで停止（LLM処理はスキップ）")
        # 最初の10セグメントを表示してトランスクリプトを確認
        for seg in transcript.segments[:10]:
            print(f"  {seg.to_timestamp()}  {seg.text.strip()}")
        return

    # -------------------------------------------------------
    # Step 3: セグメント境界検出 + 種別分類
    # -------------------------------------------------------
    # TODO: Phase 1 Step 3 で実装

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
