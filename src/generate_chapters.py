"""
src/generate_chapters.py
────────────────────────────────────────────────────────────
CONTENT セグメントのトランスクリプトから
Gemini Flash を使って YouTube チャプタータイトルを生成するモジュール

【設計方針】
全 CONTENT セグメントを1リクエストで送って JSON で受け取る。
→ API コールを最小化（15分動画で 1〜2 回程度）
→ LLM にセクション全体の文脈を渡せるため、より適切なタイトルになる

【対応プロバイダ】
  gemini   : google-generativeai（無料枠: 1500req/日）  ← デフォルト
  openai   : gpt-4o-mini（有料、ほぼゼロコスト）
  anthropic: claude-3-5-haiku（有料）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from loguru import logger

from src.config import Config
from src.detect_boundaries import DetectedSegment


# ─────────────────────────────────────────
# 返り値の型定義
# ─────────────────────────────────────────

@dataclass
class Chapter:
    """YouTube チャプター1件"""
    start_sec: float
    title:     str

    @property
    def timestamp(self) -> str:
        """YouTube 形式（H:MM:SS / M:SS）"""
        total = int(self.start_sec)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


# ─────────────────────────────────────────
# プロンプト
# ─────────────────────────────────────────

_SYSTEM_PROMPT = """\
あなたは動画編集のアシスタントです。
動画のトランスクリプト（各セクションの書き起こし）が与えられます。
各セクションに対して、YouTube の概要欄に掲載するチャプタータイトルを作成してください。

【ルール】
- タイトルは日本語で 20 文字以内
- 内容を端的に表す名詞句または体言止め（「〜の発表」「質疑応答」など）
- 接続詞や助詞で始めない
- セクションの内容が読み取れない場合は「（内容不明）」とする
- 返答は必ず以下の JSON 形式のみ。説明文・コードブロックは不要:
  [{"index": 0, "title": "タイトル"}, {"index": 1, "title": "タイトル"}, ...]
"""


def _build_user_prompt(segments: list[DetectedSegment]) -> str:
    lines = ["以下のセクションのタイトルを生成してください:\n"]
    for i, seg in enumerate(segments):
        ts = seg.to_timestamp()
        # 長すぎるトランスクリプトは先頭 400 文字に切り詰め
        text = seg.transcript_text[:400].strip()
        lines.append(f"[セクション {i}] 開始: {ts}\n{text}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────

def generate_chapters(
    segments: list[DetectedSegment],
    cfg: Config,
) -> list[Chapter]:
    """
    CONTENT セグメントのリストからチャプタータイトルを生成する。

    Args:
        segments: detect_boundaries() の返り値のうち kind=CONTENT のもの
        cfg:      設定オブジェクト

    Returns:
        Chapter のリスト（時刻昇順）
    """
    if not segments:
        logger.warning("CONTENT セグメントが空です")
        return []

    logger.info(
        f"チャプタータイトル生成開始: {len(segments)}セクション / "
        f"プロバイダ: {cfg.llm_provider}"
    )

    if cfg.llm_provider == "gemini":
        titles = _call_gemini(segments, cfg)
    elif cfg.llm_provider == "openai":
        titles = _call_openai(segments, cfg)
    elif cfg.llm_provider == "anthropic":
        titles = _call_anthropic(segments, cfg)
    else:
        raise ValueError(f"未対応の llm_provider: {cfg.llm_provider}")

    # タイトルと開始時刻を紐付けて Chapter に変換
    chapters = []
    for i, seg in enumerate(segments):
        title = titles.get(i, "（タイトル生成失敗）")
        chapters.append(Chapter(start_sec=seg.start, title=title))
        logger.debug(f"  [{seg.to_timestamp()}] {title}")

    logger.success(f"チャプタータイトル生成完了: {len(chapters)}件")
    return chapters


# ─────────────────────────────────────────
# 各プロバイダの実装
# ─────────────────────────────────────────

def _call_gemini(segments: list[DetectedSegment], cfg: Config) -> dict[int, str]:
    """Gemini Flash を呼び出してタイトルを取得する（無料枠対応）"""
    from google import genai                          # 遅延インポート
    from google.genai import types

    if not cfg.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY が設定されていません。"
            ".env に GEMINI_API_KEY=your_key を追加してください。\n"
            "取得: https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=cfg.gemini_api_key)
    user_prompt = _build_user_prompt(segments)
    logger.info(f"  Gemini API 呼び出し中... ({cfg.gemini_model})")

    response = client.models.generate_content(
        model=cfg.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=cfg.llm_temperature,
            response_mime_type="application/json",
        ),
    )
    return _parse_json_response(response.text, len(segments))


def _call_openai(segments: list[DetectedSegment], cfg: Config) -> dict[int, str]:
    """OpenAI GPT-4o-mini を呼び出してタイトルを取得する"""
    import openai  # 遅延インポート

    client = openai.OpenAI(api_key=cfg.openai_api_key)
    user_prompt = _build_user_prompt(segments)
    logger.info(f"  OpenAI API 呼び出し中... ({cfg.openai_model})")

    response = client.chat.completions.create(
        model=cfg.openai_model,
        temperature=cfg.llm_temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return _parse_json_response(response.choices[0].message.content, len(segments))


def _call_anthropic(segments: list[DetectedSegment], cfg: Config) -> dict[int, str]:
    """Anthropic Claude を呼び出してタイトルを取得する"""
    import anthropic  # 遅延インポート

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    user_prompt = _build_user_prompt(segments)
    logger.info(f"  Anthropic API 呼び出し中... ({cfg.anthropic_model})")

    response = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=1024,
        temperature=cfg.llm_temperature,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return _parse_json_response(response.content[0].text, len(segments))


# ─────────────────────────────────────────
# JSON パース（フォールバック付き）
# ─────────────────────────────────────────

def _parse_json_response(raw: str, expected_count: int) -> dict[int, str]:
    """
    LLM の返答から {index: title} 辞書を取り出す。

    LLM が ```json ... ``` でラップして返すケース、
    配列の代わりに {"titles": [...]} で返すケースに対応する。
    """
    # コードブロック除去
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSONパース失敗: {e}\n  raw: {raw[:200]}")
        # フォールバック: ゼロ埋めで返す
        return {i: "（タイトル生成失敗）" for i in range(expected_count)}

    # [{"index": 0, "title": "..."}, ...] 形式
    if isinstance(parsed, list):
        return {item["index"]: item["title"] for item in parsed if "index" in item and "title" in item}

    # {"titles": [...]} 形式
    if isinstance(parsed, dict) and "titles" in parsed:
        return {i: t for i, t in enumerate(parsed["titles"])}

    # {"0": "...", "1": "..."} 形式
    if isinstance(parsed, dict):
        return {int(k): v for k, v in parsed.items() if str(k).isdigit()}

    logger.warning(f"想定外のJSON構造: {str(parsed)[:200]}")
    return {i: "（タイトル生成失敗）" for i in range(expected_count)}
