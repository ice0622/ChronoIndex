"""
src/detect_boundaries.py
────────────────────────────────────────────────────────────
トランスクリプトからセグメント境界を検出するモジュール

検出手法:
  ① 言語パターンマッチング（司会フレーズ・定型表現）
  ② 沈黙区間検出（Whisper VAD後もセグメント間ギャップに沈黙が現れる）

セグメント種別:
  CONTENT  発表・講演などの本コンテンツ区間
  BLANK    待機・休憩・技術トラブルなどの空白区間

空白区間の扱い（方針C）:
  BLANK セグメントはチャプターリストに載せず、
  後続 CONTENT セグメントの開始時刻として吸収する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from src.config import Config
from src.transcribe import Transcript, TranscriptSegment


# ─────────────────────────────────────────
# 定数: 境界検出パターン
# ─────────────────────────────────────────

# 話者交代・セクション区切りを示す言語パターン
BOUNDARY_PATTERNS: list[str] = [
    r"それでは.{0,20}(セッション|時間|部|発表).{0,10}(開始|始め)",
    r"続きまして.{0,20}登壇者は",
    r"ありがとうございました",
    r"質疑応答.{0,10}(の時間)?を?始め",
    r"以上で.{0,20}終了",
    r"休憩",
    r"開会",
    r"閉会",
    r"では.{0,10}(次|続い)",
    r"(ご|お)紹介",
    r"拍手",
]

# 空白区間を示す言語パターン（待機・中断など）
BLANK_PATTERNS: list[str] = [
    r"しばらくお待ちください",
    r"(準備|調整)(中|を行って)",
    r"(開始|再開).{0,10}(まで|を).{0,10}お待ち",
    r"只今.{0,10}準備",
    r"間もなく.{0,10}始",
    r"休憩(時間)?.{0,10}(に入|します|中)",
    r"ただいま.{0,10}休憩",
    r"しばらく",
]

# デフォルト閾値
_SILENCE_BOUNDARY_SEC = 3.0   # これ以上のギャップ → 境界候補
_MERGE_WINDOW_SEC     = 1.0   # 近接候補のマージ幅


# ─────────────────────────────────────────
# 型定義
# ─────────────────────────────────────────

class SegmentKind(str, Enum):
    """
    チャンクの種別。
    str を継承しているので JSON シリアライズや print がそのまま使える。
    """
    CONTENT = "CONTENT"   # 発表・講演などの本コンテンツ
    BLANK   = "BLANK"     # 待機・休憩・技術トラブルなどの空白


@dataclass
class DetectedSegment:
    """
    境界で区切られた1チャンク（TranscriptSegment より大きな粒度）。

    【dataclass の合成設計】
    小さな TranscriptSegment を集約して、より意味のある単位を表現する。
    直接継承するより「list を持つ」設計のほうが変更に強い。
    """
    start:           float                    # 開始時刻（秒）
    end:             float                    # 終了時刻（秒）
    kind:            SegmentKind              # CONTENT / BLANK
    reasons:         list[str] = field(default_factory=list)
    transcript_text: str       = ""           # 含まれるテキスト先頭200文字（確認用）

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_timestamp(self) -> str:
        """開始時刻を YouTube 形式（H:MM:SS）に変換"""
        return _sec_to_ts(self.start)


# ─────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────

def detect_boundaries(transcript: Transcript, cfg: Config) -> list[DetectedSegment]:
    """
    トランスクリプトからセグメント境界を検出して分類する。

    Args:
        transcript: transcribe() の返り値
        cfg:        設定オブジェクト

    Returns:
        DetectedSegment のリスト（時刻昇順・CONTENT + BLANK 混在）
    """
    if not transcript.segments:
        logger.warning("トランスクリプトが空です")
        return []

    logger.info("境界検出開始")
    logger.info(
        f"  セグメント数: {len(transcript.segments)} / "
        f"総時間: {transcript.duration_sec / 60:.1f}分"
    )

    # ① 境界候補の収集（言語パターン + 沈黙ギャップ）
    boundary_times = _collect_boundary_times(transcript.segments, cfg)
    logger.info(f"  境界候補: {len(boundary_times)}件")

    # ② 境界でチャンクに分割
    chunks = _split_into_chunks(transcript.segments, boundary_times)

    # ③ CONTENT / BLANK に分類
    detected = _classify_chunks(chunks, cfg)

    content_n = sum(1 for d in detected if d.kind == SegmentKind.CONTENT)
    blank_n   = sum(1 for d in detected if d.kind == SegmentKind.BLANK)
    logger.success(
        f"境界検出完了: {len(detected)}チャンク "
        f"(CONTENT={content_n}, BLANK={blank_n})"
    )
    return detected


# ─────────────────────────────────────────
# 内部実装
# ─────────────────────────────────────────

def _collect_boundary_times(
    segments: list[TranscriptSegment],
    cfg: Config,
) -> list[tuple[float, str]]:
    """
    境界候補の (時刻, 理由) リストを返す。

    【なぜ候補リストにするか】
    複数の検出手法をあとから追加・削除しやすい設計にするため。
    「言語パターンが30秒に & 沈黙が29.8秒に」→ マージして1件に統合できる。
    """
    candidates: list[tuple[float, str]] = []

    for i, seg in enumerate(segments):
        # ── 言語パターンマッチング ──
        for pattern in BOUNDARY_PATTERNS:
            if re.search(pattern, seg.text):
                # そのセグメントが終わった直後を境界として記録
                candidates.append((seg.end, f"lang:{pattern[:30]}"))
                break  # 1セグメント1回だけカウント

        # ── セグメント間の沈黙ギャップ検出 ──
        # Whisper は vad_filter=True で無音を自動スキップしているため
        # 隣接セグメントの gap = seg[i+1].start - seg[i].end が実際の沈黙時間
        if i + 1 < len(segments):
            gap = segments[i + 1].start - seg.end
            if gap >= _SILENCE_BOUNDARY_SEC:
                candidates.append((seg.end, f"silence:{gap:.1f}s"))

    return _merge_close_candidates(candidates, _MERGE_WINDOW_SEC)


def _merge_close_candidates(
    candidates: list[tuple[float, str]],
    merge_window_sec: float,
) -> list[tuple[float, str]]:
    """
    1秒以内の近接候補を先頭の1件に統合する。
    例: (30.2, "lang:...") + (30.8, "silence:5.0s") → (30.2, "lang:...+silence:5.0s")
    """
    if not candidates:
        return []

    sorted_cands = sorted(candidates)
    merged: list[tuple[float, str]] = [sorted_cands[0]]

    for time, reason in sorted_cands[1:]:
        prev_time, prev_reason = merged[-1]
        if time - prev_time <= merge_window_sec:
            merged[-1] = (prev_time, f"{prev_reason}+{reason}")
        else:
            merged.append((time, reason))

    return merged


def _split_into_chunks(
    segments: list[TranscriptSegment],
    boundary_times: list[tuple[float, str]],
) -> list[dict]:
    """
    境界タイムスタンプでトランスクリプトをチャンクに分割する。

    返り値の各要素:
        {
            "segments":        list[TranscriptSegment],
            "boundary_reason": str,   # このチャンクを作った境界の理由
        }

    【アルゴリズム】
    境界時刻の直後から始まる TranscriptSegment を新チャンクの先頭とする。
    「境界の直後」= 境界時刻から 2 秒以内に開始する次のセグメント。
    """
    if not boundary_times:
        return [{"segments": segments, "boundary_reason": "no_boundary"}]

    # (境界時刻, 理由) のリストを辞書に変換
    # 一度消費したら削除するので copy を使う
    pending: list[tuple[float, str]] = sorted(boundary_times)

    chunks: list[dict] = []
    current_segs: list[TranscriptSegment] = []
    current_reason = "start"

    for seg in segments:
        # このセグメントの開始が、未消費の境界を跨いでいるか？
        while pending and seg.start >= pending[0][0] - 0.5:
            boundary_t, boundary_r = pending.pop(0)
            if current_segs:
                chunks.append({"segments": current_segs, "boundary_reason": current_reason})
                current_segs = []
            current_reason = boundary_r

        current_segs.append(seg)

    if current_segs:
        chunks.append({"segments": current_segs, "boundary_reason": current_reason})

    return chunks


def _classify_chunks(
    chunks: list[dict],
    cfg: Config,
) -> list[DetectedSegment]:
    """
    各チャンクを CONTENT / BLANK に分類して DetectedSegment のリストを返す。

    BLANK 判定条件（どちらかが真なら BLANK）:
      A) テキストに BLANK_PATTERNS がマッチする
      B) セグメント間の最大沈黙 >= cfg.blank_min_silence_sec かつ 発話密度が低い
    """
    results: list[DetectedSegment] = []

    for chunk in chunks:
        segs = chunk["segments"]
        if not segs:
            continue

        start         = segs[0].start
        end           = segs[-1].end
        combined_text = " ".join(s.text.strip() for s in segs)
        reasons       = [chunk["boundary_reason"]]

        # 判定 A: テキストパターン
        text_is_blank = any(re.search(p, combined_text) for p in BLANK_PATTERNS)

        # 判定 B: 沈黙 + 発話密度
        max_gap = max(
            (segs[i + 1].start - segs[i].end for i in range(len(segs) - 1)),
            default=0.0,
        )
        duration      = end - start
        speech_ratio  = sum(s.duration for s in segs) / duration if duration > 0 else 0.0
        silence_is_blank = (max_gap >= cfg.blank_min_silence_sec) or (
            speech_ratio < 0.15 and len(segs) <= 3
        )

        if text_is_blank:
            kind = SegmentKind.BLANK
            reasons.append("blank_text_pattern")
        elif silence_is_blank:
            kind = SegmentKind.BLANK
            reasons.append(f"low_speech(ratio={speech_ratio:.2f},max_gap={max_gap:.1f}s)")
        else:
            kind = SegmentKind.CONTENT

        result = DetectedSegment(
            start=start,
            end=end,
            kind=kind,
            reasons=reasons,
            transcript_text=combined_text[:200],
        )
        results.append(result)
        logger.debug(
            f"  [{result.to_timestamp()} → {_sec_to_ts(end)}] "
            f"{kind.value} ({len(segs)}seg, {duration / 60:.1f}min) "
            f"| {', '.join(reasons)}"
        )

    return results


# ─────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────

def _sec_to_ts(sec: float) -> str:
    """秒数を H:MM:SS / M:SS 形式に変換"""
    total = int(sec)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
