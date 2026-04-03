"""
src/diarize.py
────────────────────────────────────────────────────────────
話者ダイアライゼーションモジュール

pyannote.audio を使って音声ファイル内の話者区間を検出する。
結果は SQLite にキャッシュして、再実行時に高速化する。

【前提条件】
- .env に HUGGINGFACE_TOKEN=hf_... を設定済みであること
- pyannote/speaker-diarization-3.1 の利用規約に同意済みであること
  https://huggingface.co/pyannote/speaker-diarization-3.1
- pyannote/segmentation-3.0 の利用規約に同意済みであること
  https://huggingface.co/pyannote/segmentation-3.0
- PyTorch（CPU版）が入っていること
  pip install -r requirements-local.txt

【PHPer 向け解説: なぜ遅延インポートを使うか】
pyannote.audio は import するだけで PyTorch をロードする。
PyTorch は起動時に数百MBをメモリに展開するため、
ダイアライゼーション不要な実行時にもメモリを食ってしまう。
→ この関数が呼ばれた時だけ import することで無駄を省く。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.config import Config
from src.transcript_cache import TranscriptCache


# ─────────────────────────────────────────
# 返り値の型定義
# ─────────────────────────────────────────
@dataclass
class DiarizationSegment:
    """話者ダイアライゼーションの1区間"""
    start:   float   # 開始時刻（秒）
    end:     float   # 終了時刻（秒）
    speaker: str     # 話者ID（例: "SPEAKER_00"）


# ─────────────────────────────────────────
# SQLite スキーマ
# ─────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS diarization_cache (
    audio_id      TEXT NOT NULL,   -- 音声ファイルの指紋（TranscriptCache.make_audio_id と共通）
    segments_json TEXT NOT NULL,   -- ダイアライゼーション結果（JSON文字列）
    created_at    TEXT NOT NULL,   -- 処理日時
    PRIMARY KEY (audio_id)
)
"""


# ─────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────
def diarize(audio_path: Path, cfg: Config) -> list[DiarizationSegment]:
    """
    音声ファイルの話者ダイアライゼーションを実行する。
    SQLite キャッシュがあればモデルを実行せずに返す。

    Args:
        audio_path: 音声ファイルのパス（wav / mp3 など）
        cfg:        設定オブジェクト（HF トークン・話者数など）

    Returns:
        list[DiarizationSegment]: 時刻順に並んだ話者区間のリスト

    Raises:
        ValueError:  HUGGINGFACE_TOKEN が未設定
        ImportError: pyannote.audio が未インストール
    """
    if not cfg.huggingface_token:
        raise ValueError(
            "HUGGINGFACE_TOKEN が未設定です。\n"
            ".env に HUGGINGFACE_TOKEN=hf_... を追加してください。\n"
            "取得: https://huggingface.co/settings/tokens\n"
            "利用申請（無料・必須）:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0"
        )

    # ─── キャッシュを開く ───
    db_path = cfg.data_dir / "transcripts" / "diarization_cache.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()

    audio_id = TranscriptCache.make_audio_id(audio_path)
    logger.info(f"  audio_id: {audio_id[:8]}...（ダイアライゼーションキャッシュキー）")

    # ─── キャッシュ HIT チェック ───
    row = conn.execute(
        "SELECT segments_json FROM diarization_cache WHERE audio_id = ?",
        (audio_id,),
    ).fetchone()

    if row:
        logger.info("  ダイアライゼーション: キャッシュ HIT → モデル実行をスキップ")
        segments_data = json.loads(row[0])
        conn.close()
        return [DiarizationSegment(**s) for s in segments_data]

    # ─── pyannote.audio でモデル実行 ───
    logger.info("  ダイアライゼーション: pyannote.audio モデルを実行中...")
    logger.info("  （初回はモデルダウンロードで数分かかる場合があります）")

    try:
        from pyannote.audio import Pipeline  # 遅延インポート（起動時のメモリ節約）
    except ImportError:
        raise ImportError(
            "pyannote.audio が見つかりません。\n"
            "以下を実行してインストールしてください:\n"
            "  pip install -r requirements-local.txt\n"
            "  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu"
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=cfg.huggingface_token,
    )

    # num_speakers が指定されていれば渡す（精度が上がる）
    kwargs: dict = {}
    if cfg.diarize_num_speakers:
        kwargs["num_speakers"] = cfg.diarize_num_speakers
        logger.info(f"  話者数ヒント: {cfg.diarize_num_speakers}人")

    diarization_result = pipeline(str(audio_path), **kwargs)

    result: list[DiarizationSegment] = [
        DiarizationSegment(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in diarization_result.itertracks(yield_label=True)
    ]

    # 時刻順にソート
    result.sort(key=lambda s: s.start)

    num_speakers = len({s.speaker for s in result})
    logger.success(f"  ダイアライゼーション完了: {len(result)}区間 / {num_speakers}話者")

    # ─── キャッシュ保存 ───
    segments_json = json.dumps(
        [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in result]
    )
    conn.execute(
        "INSERT INTO diarization_cache (audio_id, segments_json, created_at) VALUES (?, ?, ?)",
        (audio_id, segments_json, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    return result


# ─────────────────────────────────────────
# 話者ラベルのマージ
# ─────────────────────────────────────────
def merge_speakers(
    transcript_segments: list,          # list[TranscriptSegment]（循環 import 回避のため list）
    diarization: list[DiarizationSegment],
) -> list:
    """
    文字起こしセグメントに話者ラベルをマージする。

    アルゴリズム:
        1. 各 TranscriptSegment の中間点（mid = (start + end) / 2）を計算
        2. mid が含まれる DiarizationSegment を探して speaker を取得
        3. 含まれるものがなければ最も近い DiarizationSegment の話者を使う

    Args:
        transcript_segments: TranscriptSegment のリスト（in-place で書き換える）
        diarization:         DiarizationSegment のリスト

    Returns:
        同じ transcript_segments リスト（speaker フィールドを更新済み）
    """
    if not diarization:
        return transcript_segments

    for seg in transcript_segments:
        mid = (seg.start + seg.end) / 2.0
        seg.speaker = _find_speaker_at(mid, diarization)

    return transcript_segments


def _find_speaker_at(time_sec: float, diarization: list[DiarizationSegment]) -> str:
    """
    指定時刻を含む話者区間の話者IDを返す。
    含まれる区間が複数あれば最初のもの。
    1つもなければ最も近い区間の話者を返す。

    【計算量について】
    要素数が数千程度なので線形探索で十分。
    パフォーマンスが問題になる場合は bisect で改善できる。
    """
    # パス1: 完全に含む区間を探す
    for seg in diarization:
        if seg.start <= time_sec <= seg.end:
            return seg.speaker

    # パス2: 含まれなければ最も近い区間
    closest = min(
        diarization,
        key=lambda s: min(abs(s.start - time_sec), abs(s.end - time_sec)),
    )
    return closest.speaker
