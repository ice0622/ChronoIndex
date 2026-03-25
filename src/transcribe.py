"""
src/transcribe.py
────────────────────────────────────────────────────────────
音声ファイルをWhisperで文字起こしするモジュール

【Python設計思想 ⑤: イテレータ / ジェネレータ】
Whisperは音声を「セグメント（断片）」の連続として返す。
1時間の音声なら数千件のセグメントが出てくる。
これを全部リストに溜め込むとメモリを大量消費する。

→ ジェネレータ（yield）を使うと「1件処理したら1件渡す」流れ作業になり
  メモリ効率が大幅に改善する。

【Python設計思想 ⑥: dataclass の継承より合成】
TranscriptSegment という小さな型を作って
それをリストで持つ Transcript にまとめる。
「小さい型を組み合わせる」設計がPythonらしい。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.config import Config


# ─────────────────────────────────────────
# 返り値の型定義
# ─────────────────────────────────────────
@dataclass
class TranscriptSegment:
    """
    1つの発話断片（セグメント）。
    Whisperは音声を数秒〜数十秒のブロックに分割してテキスト化する。
    """
    start: float   # 開始時刻（秒）
    end: float     # 終了時刻（秒）
    text: str      # 認識されたテキスト
    speaker: str = "UNKNOWN"  # 話者ID（話者分離を実装したら埋まる）

    @property
    def duration(self) -> float:
        """
        【property デコレータ】
        メソッドなのに属性のようにアクセスできる。
        segment.duration() ではなく segment.duration と書ける。
        「計算で求まる値」をプロパティにするのがPythonらしい書き方。
        """
        return self.end - self.start

    def to_timestamp(self) -> str:
        """開始時刻をYouTube形式（H:MM:SS）の文字列に変換する"""
        total = int(self.start)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


@dataclass
class Transcript:
    """
    動画全体の文字起こし結果。
    segments のリストと言語情報を持つ。
    """
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "ja"
    audio_path: Path = field(default_factory=lambda: Path(""))

    @property
    def full_text(self) -> str:
        """全セグメントのテキストを結合して返す"""
        return " ".join(s.text.strip() for s in self.segments)

    @property
    def duration_sec(self) -> float:
        """音声全体の長さ（最後のセグメントの終了時刻）"""
        if not self.segments:
            return 0.0
        return self.segments[-1].end


# ─────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────
def transcribe(audio_path: Path, cfg: Config) -> Transcript:
    """
    音声ファイルをWhisperで文字起こしする。

    Args:
        audio_path: 音声ファイルのパス（mp3 / wav など）
        cfg:        設定オブジェクト（モデルサイズ・言語など）

    Returns:
        Transcript: タイムスタンプ付きの文字起こし結果
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    logger.info(f"文字起こし開始: {audio_path.name}")
    logger.info(f"  モード: {cfg.asr_mode} / モデル: {cfg.whisper_model_size} / 言語: {cfg.audio_language}")

    if cfg.asr_mode == "api":
        return _transcribe_via_api(audio_path, cfg)
    else:
        return _transcribe_local(audio_path, cfg)


# ─────────────────────────────────────────
# ローカルWhisper（faster-whisper）
# ─────────────────────────────────────────
def _transcribe_local(audio_path: Path, cfg: Config) -> Transcript:
    """
    faster-whisper をローカルで実行して文字起こしする。
    初回はモデルをダウンロードするため数分かかる場合がある。

    【遅延インポート（Lazy Import）】
    faster_whisper は import するだけでメモリを消費する重いライブラリ。
    この関数が呼ばれた時だけインポートすることで、
    api モード使用時のメモリ浪費を避けられる。
    """
    from faster_whisper import WhisperModel  # ← 遅延インポート

    logger.info(f"  モデルをロード中... ({cfg.whisper_model_size})")
    model = WhisperModel(
        cfg.whisper_model_size,
        device="cpu",          # GPUなし環境
        compute_type="int8",   # CPU向けの量子化（速度改善・精度はほぼ同じ）
    )
    logger.info("  モデルロード完了")

    # segments はジェネレータ（遅延評価）で返ってくる
    # → list() で全件取得するまで実際の処理は走らない
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=cfg.audio_language,
        beam_size=5,           # ビームサーチの幅（大きいほど精度↑・速度↓）
        vad_filter=True,       # 無音区間を自動でスキップ（VAD: Voice Activity Detection）
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(f"  検出言語: {info.language} (確信度: {info.language_probability:.1%})")

    # ─── ジェネレータを消費しながら進捗表示 ───
    # 【tqdm を使わず loguru で進捗を出す簡易実装】
    result_segments: list[TranscriptSegment] = []
    for i, seg in enumerate(segments_gen):
        # 【enumerate とは】
        # for i, item in enumerate(iterable) でインデックス付きループ。
        # range(len(list)) を使うより Pythonic。
        segment = TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
        )
        result_segments.append(segment)

        # 100件ごとに進捗をログ出力
        if (i + 1) % 100 == 0:
            logger.info(f"  処理済み: {i+1}件 / 現在位置: {segment.to_timestamp()}")

    transcript = Transcript(
        segments=result_segments,
        language=info.language,
        audio_path=audio_path,
    )
    logger.success(
        f"文字起こし完了: {len(result_segments)}セグメント / "
        f"総時間: {transcript.duration_sec/60:.1f}分"
    )
    return transcript


# ─────────────────────────────────────────
# OpenAI Whisper API（有料・高速）
# ─────────────────────────────────────────
def _transcribe_via_api(audio_path: Path, cfg: Config) -> Transcript:
    """
    OpenAI Whisper API を使って文字起こしする。
    コスト: $0.006/分（1時間 ≒ 52円）
    ローカルより10〜20倍速いが費用がかかる。
    """
    import openai  # ← 遅延インポート

    client = openai.OpenAI(api_key=cfg.openai_api_key)

    # OpenAI API のファイルサイズ制限は 25MB
    # 超える場合は分割が必要（今後の課題）
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 24:
        raise ValueError(
            f"ファイルサイズ {file_size_mb:.1f}MB が API 制限（25MB）を超えています。"
            "ローカルモード（ASR_MODE=local）を使用するか、音声を分割してください。"
        )

    logger.info(f"  OpenAI Whisper API に送信中... ({file_size_mb:.1f}MB)")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=cfg.audio_language,
            response_format="verbose_json",  # タイムスタンプ付きで取得
            timestamp_granularities=["segment"],
        )

    result_segments = [
        TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
        )
        for seg in response.segments  # ← リスト内包表記
    ]

    transcript = Transcript(
        segments=result_segments,
        language=response.language,
        audio_path=audio_path,
    )
    logger.success(f"API文字起こし完了: {len(result_segments)}セグメント")
    return transcript
