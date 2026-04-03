"""
evaluate/test_diarize.py
────────────────────────────────────────────────────────────
話者ダイアライゼーションモジュールの単体テスト

テスト方針:
- pyannote.audio や HuggingFace モデルは呼ばない（モック不要・オフライン実行可能）
- _find_speaker_at と merge_speakers のロジックを集中的にテスト
- TranscriptSegment の speaker フィールドへの書き込みを検証

実行方法:
    docker compose run --rm dev pytest evaluate/test_diarize.py -v
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from src.diarize import DiarizationSegment, _find_speaker_at, merge_speakers
from src.transcribe import TranscriptSegment


# ─────────────────────────────────────────
# フィクスチャ
# ─────────────────────────────────────────
@pytest.fixture
def diarization_simple() -> list[DiarizationSegment]:
    """
    単純な2話者のダイアライゼーション結果。

    0─────────10────────20────────30
    |  SP_00  |  SP_01  |  SP_00  |
    """
    return [
        DiarizationSegment(start=0.0,  end=10.0, speaker="SPEAKER_00"),
        DiarizationSegment(start=10.0, end=20.0, speaker="SPEAKER_01"),
        DiarizationSegment(start=20.0, end=30.0, speaker="SPEAKER_00"),
    ]


@pytest.fixture
def diarization_with_gap() -> list[DiarizationSegment]:
    """
    話者区間の間に「無音ギャップ」がある場合。

    0─────5  (gap)  10────20
    | SP_00 |      | SP_01 |
    """
    return [
        DiarizationSegment(start=0.0,  end=5.0,  speaker="SPEAKER_00"),
        DiarizationSegment(start=10.0, end=20.0, speaker="SPEAKER_01"),
    ]


# ─────────────────────────────────────────
# _find_speaker_at のテスト
# ─────────────────────────────────────────
class TestFindSpeakerAt:
    def test_exact_start(self, diarization_simple):
        """区間の開始点ぴったりの場合"""
        assert _find_speaker_at(0.0, diarization_simple) == "SPEAKER_00"
        assert _find_speaker_at(10.0, diarization_simple) in ("SPEAKER_00", "SPEAKER_01")

    def test_exact_end(self, diarization_simple):
        """区間の終了点ぴったりの場合"""
        assert _find_speaker_at(30.0, diarization_simple) == "SPEAKER_00"

    def test_midpoint(self, diarization_simple):
        """区間の中間点"""
        assert _find_speaker_at(5.0, diarization_simple) == "SPEAKER_00"
        assert _find_speaker_at(15.0, diarization_simple) == "SPEAKER_01"
        assert _find_speaker_at(25.0, diarization_simple) == "SPEAKER_00"

    def test_gap_falls_back_to_nearest(self, diarization_with_gap):
        """
        ギャップ内の時刻（7.5秒）は、最も近い区間の話者を返す。
        SP_00 の end=5.0 → 距離 2.5
        SP_01 の start=10.0 → 距離 2.5
        最小距離が同じ場合は実装依存だが、いずれかの話者になる。
        """
        speaker = _find_speaker_at(7.5, diarization_with_gap)
        assert speaker in ("SPEAKER_00", "SPEAKER_01")

    def test_gap_closer_to_first(self, diarization_with_gap):
        """ギャップ内でも SP_00 に近い点（5.1秒）は SP_00"""
        assert _find_speaker_at(5.1, diarization_with_gap) == "SPEAKER_00"

    def test_gap_closer_to_second(self, diarization_with_gap):
        """ギャップ内でも SP_01 に近い点（9.9秒）は SP_01"""
        assert _find_speaker_at(9.9, diarization_with_gap) == "SPEAKER_01"

    def test_before_first_segment(self, diarization_with_gap):
        """最初の区間より前の時刻（負の値）は最初の話者"""
        assert _find_speaker_at(-1.0, diarization_with_gap) == "SPEAKER_00"

    def test_after_last_segment(self, diarization_with_gap):
        """最後の区間より後の時刻は最後の話者"""
        assert _find_speaker_at(999.0, diarization_with_gap) == "SPEAKER_01"

    def test_single_segment(self):
        """ダイアライゼーションが1区間だけの場合は常にその話者"""
        diarization = [DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00")]
        assert _find_speaker_at(30.0, diarization) == "SPEAKER_00"
        assert _find_speaker_at(999.0, diarization) == "SPEAKER_00"


# ─────────────────────────────────────────
# merge_speakers のテスト
# ─────────────────────────────────────────
class TestMergeSpeakers:
    def _make_segments(self, data: list[tuple[float, float, str]]) -> list[TranscriptSegment]:
        """(start, end, text) のリストから TranscriptSegment を作るヘルパー"""
        return [
            TranscriptSegment(start=s, end=e, text=t)
            for s, e, t in data
        ]

    def test_basic_merge(self, diarization_simple):
        """中間点が正しい話者区間に含まれる基本ケース"""
        segments = self._make_segments([
            (0.0,  8.0,  "こんにちは"),   # mid=4.0  → SPEAKER_00
            (10.5, 18.0, "ありがとう"),   # mid=14.25 → SPEAKER_01
            (21.0, 29.0, "さようなら"),   # mid=25.0  → SPEAKER_00
        ])
        result = merge_speakers(segments, diarization_simple)

        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"
        assert result[2].speaker == "SPEAKER_00"

    def test_empty_diarization_returns_unchanged(self):
        """ダイアライゼーションが空の場合はセグメントをそのまま返す"""
        segments = self._make_segments([(0.0, 5.0, "テスト")])
        result = merge_speakers(segments, [])

        assert result[0].speaker == "UNKNOWN"  # 変化なし

    def test_empty_segments(self, diarization_simple):
        """セグメントが空の場合は空リストを返す"""
        result = merge_speakers([], diarization_simple)
        assert result == []

    def test_returns_same_list(self, diarization_simple):
        """in-place 変更 + 同じリストを返すことを確認"""
        segments = self._make_segments([(0.0, 5.0, "テスト")])
        result = merge_speakers(segments, diarization_simple)
        assert result is segments  # 同じオブジェクト

    def test_unknown_overwritten(self, diarization_simple):
        """初期値 UNKNOWN が上書きされることを確認"""
        seg = TranscriptSegment(start=5.0, end=9.0, text="テスト")
        assert seg.speaker == "UNKNOWN"

        merge_speakers([seg], diarization_simple)

        assert seg.speaker != "UNKNOWN"
        assert seg.speaker == "SPEAKER_00"

    def test_three_speakers(self):
        """3話者のケース"""
        diarization = [
            DiarizationSegment(start=0.0,  end=10.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=10.0, end=20.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=20.0, end=30.0, speaker="SPEAKER_02"),
        ]
        segments = self._make_segments([
            (1.0,  3.0,  "話者A"),   # mid=2.0  → SPEAKER_00
            (11.0, 13.0, "話者B"),   # mid=12.0 → SPEAKER_01
            (21.0, 23.0, "話者C"),   # mid=22.0 → SPEAKER_02
        ])
        merge_speakers(segments, diarization)

        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].speaker == "SPEAKER_01"
        assert segments[2].speaker == "SPEAKER_02"


# ─────────────────────────────────────────
# DiarizationSegment のテスト
# ─────────────────────────────────────────
class TestDiarizationSegment:
    def test_dataclass_creation(self):
        seg = DiarizationSegment(start=1.5, end=3.0, speaker="SPEAKER_00")
        assert seg.start == 1.5
        assert seg.end == 3.0
        assert seg.speaker == "SPEAKER_00"

    def test_from_dict(self):
        """JSON キャッシュから dict を経由して復元できる"""
        data = {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"}
        seg = DiarizationSegment(**data)
        assert seg.speaker == "SPEAKER_01"
