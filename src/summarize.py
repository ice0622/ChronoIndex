"""
src/summarize.py
────────────────────────────────────────────────────────────
文字起こし結果（Transcript）から動画の要約を生成するモジュール。

Gemini Flash（デフォルト）/ OpenAI / Anthropic に対応。

使い方（単体実行）:
    python -m src.summarize --transcript data/transcripts/hoge.txt
    python -m src.summarize --transcript data/transcripts/hoge.txt --output data/output/hoge_summary.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
from loguru import logger

from src.config import Config, load_config
from src.transcribe import Transcript


# ─────────────────────────────────────────
# 返り値の型定義
# ─────────────────────────────────────────

@dataclass
class Summary:
    """動画要約の結果"""
    text: str          # 要約本文
    provider: str      # 使用したLLMプロバイダ名
    model: str         # 使用したモデル名


# ─────────────────────────────────────────
# プロンプト
# ─────────────────────────────────────────

_SYSTEM_PROMPT = """\
あなたは動画コンテンツの要約を作成するアシスタントです。
YouTube動画の文字起こしテキストが与えられます。
単なる概要ではなく、「内容を再現できるレベルの高密度な要約」を作成してください。

【目的】
視聴しなくても、動画の内容・ノウハウ・具体例を理解し、再利用できる状態にする

【要約の構成】

## 概要
- 動画のテーマ・目的・対象者を2〜3文で説明

## 主なポイント
- 動画の重要な論点を箇条書きで整理（5〜10項目）
- 抽象的にせず、できるだけ具体的に書く

## 詳細（重要）
以下の観点から、内容を具体的に整理する：

- **具体的な手法・手順**
- **実例・ケーススタディ**
- **ツール・技術・用語（必要なら簡潔に補足）**
- **メリット・デメリット**
- **注意点・落とし穴**
- **発言者の主張・意見（根拠があるもの）**

※ 該当する内容がある場合のみ記載

## 結論・まとめ
- 動画の結論や、視聴後に得られる知見を1〜2文でまとめる

---

【ルール】
- 日本語で出力する
- 実際に語られた内容のみを要約し、推測や補完を行わない
- 重要な具体情報（数値・手順・固有名詞）は省略しない
- 抽象化しすぎない（例：「効率化できる」だけで終わらせない）
- 冗長な言い換えは禁止（情報密度を優先）
- Markdown形式で出力（見出し・箇条書き）

【出力の質基準】
- 元動画の「使える情報」を落とさない
- 読めば内容を他人に説明できるレベル
"""


def _build_user_prompt(transcript: Transcript) -> str:
    """Transcript から要約用プロンプトを構築する"""
    # 長すぎる場合は先頭・末尾を優先して切り詰める（最大 30,000 文字）
    full_text = transcript.full_text
    max_chars = 30_000
    if len(full_text) > max_chars:
        half = max_chars // 2
        full_text = full_text[:half] + "\n\n[... 中略 ...]\n\n" + full_text[-half:]

    return f"以下の動画の文字起こしを要約してください:\n\n{full_text}"


# ─────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────

def summarize(transcript: Transcript, cfg: Config) -> Summary:
    """
    Transcript オブジェクトから要約を生成する。

    Args:
        transcript: transcribe() の返り値
        cfg:        設定オブジェクト

    Returns:
        Summary オブジェクト
    """
    logger.info(
        f"要約生成開始: {len(transcript.segments)}セグメント / "
        f"プロバイダ: {cfg.llm_provider}"
    )

    if cfg.llm_provider == "gemini":
        text = _call_gemini(transcript, cfg)
        model = cfg.gemini_model
    elif cfg.llm_provider == "openai":
        text = _call_openai(transcript, cfg)
        model = cfg.openai_model
    elif cfg.llm_provider == "anthropic":
        text = _call_anthropic(transcript, cfg)
        model = cfg.anthropic_model
    else:
        raise ValueError(f"未対応の llm_provider: {cfg.llm_provider}")

    logger.success(f"要約生成完了 ({cfg.llm_provider} / {model})")
    return Summary(text=text, provider=cfg.llm_provider, model=model)


# ─────────────────────────────────────────
# 各プロバイダの実装
# ─────────────────────────────────────────

def _call_gemini(transcript: Transcript, cfg: Config) -> str:
    """Gemini Flash を呼び出して要約を取得する"""
    from google import genai
    from google.genai import types

    if not cfg.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY が設定されていません。\n"
            "取得: https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=cfg.gemini_api_key)
    user_prompt = _build_user_prompt(transcript)
    logger.info(f"  Gemini API 呼び出し中... ({cfg.gemini_model})")

    response = client.models.generate_content(
        model=cfg.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=cfg.llm_temperature,
        ),
    )
    return response.text.strip()


def _call_openai(transcript: Transcript, cfg: Config) -> str:
    """OpenAI を呼び出して要約を取得する"""
    from openai import OpenAI

    if not cfg.openai_api_key:
        raise ValueError("OPENAI_API_KEY が設定されていません。")

    client = OpenAI(api_key=cfg.openai_api_key)
    user_prompt = _build_user_prompt(transcript)
    logger.info(f"  OpenAI API 呼び出し中... ({cfg.openai_model})")

    response = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.llm_temperature,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(transcript: Transcript, cfg: Config) -> str:
    """Anthropic Claude を呼び出して要約を取得する"""
    import anthropic

    if not cfg.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY が設定されていません。")

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    user_prompt = _build_user_prompt(transcript)
    logger.info(f"  Anthropic API 呼び出し中... ({cfg.anthropic_model})")

    message = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=cfg.llm_temperature,
    )
    return message.content[0].text.strip()


# ─────────────────────────────────────────
# 単体実行用 CLI
# ─────────────────────────────────────────

@click.command()
@click.option("--transcript", "-t", required=True, type=click.Path(), help="文字起こしテキストファイルのパス")
@click.option("--output", "-o", default=None, type=click.Path(), help="要約の出力先ファイルパス（省略時: stdout）")
def _cli(transcript: str, output: str | None) -> None:
    """既存の文字起こしテキストから要約を生成する"""
    cfg = load_config()

    # テキストファイルを Transcript として読み込む
    txt_path = Path(transcript)
    raw_text = txt_path.read_text(encoding="utf-8")

    # タイムスタンプ行をパースして TranscriptSegment に変換する
    from src.transcribe import TranscriptSegment
    import re

    segments = []
    ts_pattern = re.compile(r"^(\d+:\d{2}(?::\d{2})?)\s+(.+)$")
    for line in raw_text.splitlines():
        m = ts_pattern.match(line.strip())
        if m:
            ts_str, text = m.group(1), m.group(2)
            # 話者ラベル "[SPEAKER_XX]" を除去
            text = re.sub(r"^\[.*?\]\s*", "", text)
            # タイムスタンプを秒数に変換
            parts = ts_str.split(":")
            if len(parts) == 3:
                sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                sec = int(parts[0]) * 60 + int(parts[1])
            segments.append(TranscriptSegment(start=float(sec), end=float(sec), text=text))
        else:
            # タイムスタンプなし行はそのままテキストとして追加
            if line.strip():
                segments.append(TranscriptSegment(start=0.0, end=0.0, text=line.strip()))

    tr = Transcript(segments=segments)
    summary = summarize(tr, cfg)

    print("\n" + "=" * 60)
    print(f"# 要約 ({summary.provider} / {summary.model})")
    print("=" * 60)
    print(summary.text)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary.text, encoding="utf-8")
        logger.success(f"要約を保存しました: {output}")


if __name__ == "__main__":
    _cli()
