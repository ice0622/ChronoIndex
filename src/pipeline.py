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
@click.option("--summarize", "-s", is_flag=True, default=False, help="文字起こし後に Gemini で要約を生成する")
@click.option("--summary-output", default=None, type=click.Path(), help="要約の出力先ファイルパス（省略時: stdout のみ）")
def main(url: str | None, audio: str | None, output: str | None, dry_run: bool, summarize: bool, summary_output: str | None) -> None:
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
        # --output 未指定時は動画タイトルをファイル名に使う
        if output is None and not dry_run:
            output = str(cfg.data_dir / "output" / f"{audio_info.title}.txt")

    # --audio 直接指定かつ --output 未指定の場合もファイル名から自動生成
    if audio and output is None and not dry_run:
        output = str(cfg.data_dir / "output" / f"{Path(audio).stem}.txt")

    # -------------------------------------------------------
    # Step 2: 音声認識（ASR）
    # -------------------------------------------------------
    logger.info(f"[Step 2] 音声認識: {audio}")
    from src.transcribe import transcribe
    transcript = transcribe(Path(audio), cfg)
    logger.info(f"  → {len(transcript.segments)}セグメント / 総時間: {transcript.duration_sec/60:.1f}分")

    # 文字起こしをテキストファイルに保存（タイムスタンプ付き）
    transcript_title = audio_info.title if url else Path(audio).stem
    transcript_txt_path = cfg.data_dir / "transcripts" / f"{transcript_title}.txt"
    transcript_txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{seg.to_timestamp()}  [{seg.speaker}] {seg.text.strip()}"
        if seg.speaker != "UNKNOWN"
        else f"{seg.to_timestamp()}  {seg.text.strip()}"
        for seg in transcript.segments
    ]
    transcript_txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  → 文字起こしテキスト保存: {transcript_txt_path}")

    if dry_run:
        logger.info("[dry-run] ここで停止（LLM処理はスキップ）")
        for seg in transcript.segments[:10]:
            print(f"  {seg.to_timestamp()}  {seg.text.strip()}")
        return

    # -------------------------------------------------------
    # Step 2.5: 要約生成（--summarize 指定時のみ）
    # -------------------------------------------------------
    if summarize:
        logger.info("[Step 2.5] 要約生成")
        from src.summarize import summarize as generate_summary
        summary = generate_summary(transcript, cfg)

        print("\n" + "=" * 60)
        print(f"# 要約 ({summary.provider} / {summary.model})")
        print("=" * 60)
        print(summary.text)

        if summary_output:
            summary_out_path = Path(summary_output)
            summary_out_path.parent.mkdir(parents=True, exist_ok=True)
            summary_out_path.write_text(summary.text, encoding="utf-8")
            logger.success(f"要約を保存しました: {summary_output}")
        elif output is not None:
            # --output が指定されていれば同じディレクトリに _summary.txt として保存
            auto_summary_path = Path(output).with_stem(Path(output).stem + "_summary")
            auto_summary_path.parent.mkdir(parents=True, exist_ok=True)
            auto_summary_path.write_text(summary.text, encoding="utf-8")
            logger.success(f"要約を保存しました: {auto_summary_path}")

    # -------------------------------------------------------
    # Step 3: セグメント境界検出 + 種別分類
    # -------------------------------------------------------
    logger.info("[Step 3] 境界検出")
    from src.detect_boundaries import detect_boundaries, SegmentKind
    detected_segments = detect_boundaries(transcript, cfg)
    content_segments = [s for s in detected_segments if s.kind == SegmentKind.CONTENT]
    logger.info(f"  → CONTENT={len(content_segments)}件 / 全{len(detected_segments)}件")

    # -------------------------------------------------------
    # Step 4: LLM によるチャプタータイトル生成
    # -------------------------------------------------------
    logger.info("[Step 4] チャプタータイトル生成")
    from src.generate_chapters import generate_chapters
    chapters = generate_chapters(content_segments, cfg)
    logger.info(f"  → {len(chapters)}チャプター生成完了")

    # -------------------------------------------------------
    # Step 5: YouTube 形式への整形・出力
    # -------------------------------------------------------
    logger.info("[Step 5] 出力")
    lines = [f"{ch.timestamp}  {ch.title}" for ch in chapters]
    result_text = "\n".join(lines)

    print("\n" + "=" * 50)
    print("# チャプターリスト（YouTube 概要欄用）")
    print("=" * 50)
    print(result_text)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result_text, encoding="utf-8")
        logger.success(f"保存完了: {output}")

    logger.success("パイプライン完了")


if __name__ == "__main__":
    main()
