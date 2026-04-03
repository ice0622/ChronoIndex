"""
evaluate/demo_diarization_output.py
────────────────────────────────────────────────────────────
ダイアライゼーション出力のデモ

pyannote.audio を呼ばずに、実際の文字起こしデータに
モックの話者ラベルを適用して「before / after」を比較する。

実行方法:
    docker compose run --rm dev python evaluate/demo_diarization_output.py
"""

from __future__ import annotations
from pathlib import Path
from src.diarize import DiarizationSegment, merge_speakers
from src.transcribe import TranscriptSegment

# ─────────────────────────────────────────
# 既存の文字起こしファイルを読み込む
# ─────────────────────────────────────────
TRANSCRIPT_PATH = Path("/app/data/transcripts/－【パーソルクロステクノロジー】プロエンジニア×本音トークセッション.txt")

def parse_transcript(path: Path) -> list[TranscriptSegment]:
    """テキストファイル（MM:SS  テキスト）を TranscriptSegment に変換"""
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            continue
        ts, text = parts
        # H:MM:SS または M:SS を秒に変換
        t = ts.strip().split(":")
        if len(t) == 3:
            sec = int(t[0]) * 3600 + int(t[1]) * 60 + float(t[2])
        else:
            sec = int(t[0]) * 60 + float(t[1])
        segments.append(TranscriptSegment(start=sec, end=sec + 5.0, text=text))
    return segments

# ─────────────────────────────────────────
# モックの話者区間（実際の動画内容に基づく）
# ─────────────────────────────────────────
# 動画: パーソルクロステクノロジー 本音トークセッション
# 登場人物: 河本浩二(SP_00) / 上田雅美(SP_01) / 沢田浩二(SP_02) / 楠剛平(SP_03)
MOCK_DIARIZATION = [
    DiarizationSegment(start=0.0,   end=2.0,   speaker="SPEAKER_00"),  # ナレーション
    DiarizationSegment(start=2.0,   end=19.0,  speaker="SPEAKER_00"),  # 河本浩二
    DiarizationSegment(start=19.0,  end=23.0,  speaker="SPEAKER_01"),  # 上田雅美
    DiarizationSegment(start=23.0,  end=27.0,  speaker="SPEAKER_02"),  # 沢田浩二
    DiarizationSegment(start=27.0,  end=34.0,  speaker="SPEAKER_03"),  # 楠剛平
    DiarizationSegment(start=34.0,  end=54.0,  speaker="SPEAKER_00"),  # 会社説明（ナレーション/MC）
    DiarizationSegment(start=54.0,  end=59.0,  speaker="SPEAKER_01"),
    DiarizationSegment(start=59.0,  end=111.0, speaker="SPEAKER_00"),
    DiarizationSegment(start=111.0, end=140.0, speaker="SPEAKER_03"),
    DiarizationSegment(start=140.0, end=200.0, speaker="SPEAKER_02"),
    DiarizationSegment(start=200.0, end=260.0, speaker="SPEAKER_01"),
    DiarizationSegment(start=260.0, end=360.0, speaker="SPEAKER_00"),
    DiarizationSegment(start=360.0, end=480.0, speaker="SPEAKER_02"),
    DiarizationSegment(start=480.0, end=600.0, speaker="SPEAKER_01"),
    DiarizationSegment(start=600.0, end=999.0, speaker="SPEAKER_00"),
]

# 話者IDと実名のマッピング（実運用では未設定のまま SPEAKER_00 表記になる）
SPEAKER_NAMES = {
    "SPEAKER_00": "河本浩二",
    "SPEAKER_01": "上田雅美",
    "SPEAKER_02": "沢田浩二",
    "SPEAKER_03": "楠剛平",
}

# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
def main():
    segments = parse_transcript(TRANSCRIPT_PATH)
    print(f"文字起こし読み込み: {len(segments)}セグメント")
    print()

    # ── BEFORE（話者ラベルなし）──
    print("=" * 60)
    print("【BEFORE】ダイアライゼーションなし")
    print("=" * 60)
    for seg in segments[:10]:
        ts = _sec_to_ts(seg.start)
        print(f"{ts}  {seg.text.strip()}")
    print("  ...")
    print()

    # ── merge_speakers を適用 ──
    merge_speakers(segments, MOCK_DIARIZATION)

    # ── AFTER（話者ラベルあり）──
    print("=" * 60)
    print("【AFTER】ダイアライゼーションあり（SPEAKER_XX 表記）")
    print("=" * 60)
    for seg in segments[:10]:
        ts = _sec_to_ts(seg.start)
        print(f"{ts}  [{seg.speaker}] {seg.text.strip()}")
    print("  ...")
    print()

    # ── 話者名マッピング適用後の例 ──
    print("=" * 60)
    print("【参考】話者名を手動マッピングした場合のイメージ")
    print("（SPEAKER_XX → 実名 は現時点では自動化されていない）")
    print("=" * 60)
    for seg in segments[:10]:
        ts = _sec_to_ts(seg.start)
        name = SPEAKER_NAMES.get(seg.speaker, seg.speaker)
        print(f"{ts}  [{name}] {seg.text.strip()}")
    print("  ...")
    print()

    # ── 話者分布の統計 ──
    print("=" * 60)
    print("【話者別セグメント数】")
    print("=" * 60)
    from collections import Counter
    counts = Counter(seg.speaker for seg in segments)
    for speaker, count in sorted(counts.items()):
        name = SPEAKER_NAMES.get(speaker, speaker)
        ratio = count / len(segments) * 100
        bar = "█" * (count // 3)
        print(f"  {speaker} ({name:6s}): {count:3d}件 ({ratio:5.1f}%)  {bar}")

    # ── ダイアライゼーション有効時のパイプライン出力ファイルのプレビュー ──
    print()
    print("=" * 60)
    print("【data/transcripts/*.txt への書き込みイメージ（先頭20行）】")
    print("=" * 60)
    for seg in segments[:20]:
        ts = _sec_to_ts(seg.start)
        if seg.speaker != "UNKNOWN":
            print(f"{ts}  [{seg.speaker}] {seg.text.strip()}")
        else:
            print(f"{ts}  {seg.text.strip()}")


def _sec_to_ts(sec: float) -> str:
    total = int(sec)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


if __name__ == "__main__":
    main()
