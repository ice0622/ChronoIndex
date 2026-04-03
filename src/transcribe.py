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

    if cfg.asr_mode == "groq":
        transcript = _transcribe_groq(audio_path, cfg)
    elif cfg.asr_mode == "api":
        transcript = _transcribe_via_api(audio_path, cfg)
    else:
        transcript = _transcribe_local(audio_path, cfg)

    # ─── 話者ダイアライゼーション（有効な場合のみ実行）───
    if cfg.diarization_enabled:
        logger.info("[Optional] 話者ダイアライゼーション")
        from src.diarize import diarize, merge_speakers  # 遅延インポート
        diarization = diarize(audio_path, cfg)
        merge_speakers(transcript.segments, diarization)
        logger.info(f"  → 話者ラベルを {len(transcript.segments)}セグメントにマージ完了")

    return transcript


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


# ─────────────────────────────────────────
# Groq Whisper API（無料枠・超高速 ← 推奨）
# ─────────────────────────────────────────
_GROQ_SIZE_LIMIT_MB = 24.0   # API制限 25MBの安全側
_CHUNK_MINUTES      = 10     # 10分刻み: 1チャンク=600秒、7200秒/時÷600秒=12チャンク/時
_OVERLAP_SEC        = 10     # チャンク境界の発話欠け防止用オーバーラップ


def _transcribe_groq(audio_path: Path, cfg: Config) -> Transcript:
    """
    Groq Whisper API + SQLite キャッシュで文字起こしする。

    【設計のポイント（PHPer 向け）】
    チャンクを処理するたびに DB に保存するので、
    途中でレートリミットに当たっても再実行すると続きから再開できる。

    PHP でいうとこういうイメージ:
        foreach ($chunks as $i => $chunk) {
            if (alreadyInDB($audioId, $i)) continue;  // スキップ！
            $result = callGroqAPI($chunk);
            saveToDB($audioId, $i, $result);
        }
        $allSegments = loadAllFromDB($audioId);
    """
    import re
    import subprocess
    import tempfile
    import time
    from groq import Groq, RateLimitError

    from src.transcript_cache import TranscriptCache

    if not cfg.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY が未設定です。"
            ".env に GROQ_API_KEY=xxxx を追加してください。"
            "無料登録: https://console.groq.com"
        )

    client = Groq(api_key=cfg.groq_api_key)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    # SQLite キャッシュを開く（なければ自動作成）
    db_path  = cfg.data_dir / "transcripts" / "transcript_cache.db"
    cache    = TranscriptCache(db_path)
    audio_id = TranscriptCache.make_audio_id(audio_path)

    logger.info(f"  ファイルサイズ: {file_size_mb:.1f}MB")
    logger.info(f"  audio_id: {audio_id[:8]}...（キャッシュキー）")

    # ─── ffprobe で総再生時間を取得（ファイルをメモリに読まない）───
    # pydub の AudioSegment.from_file() は全データをメモリ展開するため OOM になる。
    # ffprobe はファイルのヘッダだけ読むので軽量。
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    total_sec   = float(result.stdout.strip())
    chunk_sec   = _CHUNK_MINUTES * 60
    overlap_sec = float(_OVERLAP_SEC)

    # 総チャンク数を計算
    import math
    total_chunks = math.ceil(total_sec / chunk_sec)

    cached_count = cache.count_chunks(audio_id)
    logger.info(f"  総再生時間: {total_sec/60:.1f}分 / チャンク数: {total_chunks} / キャッシュ済み: {cached_count}")

    # ─── チャンクごとに処理 ───
    chunk_index = 0

    while chunk_index < total_chunks:
        offset_sec = chunk_index * chunk_sec

        # ── キャッシュ済みならスキップ（再実行時のメリット）──
        if cache.has_chunk(audio_id, chunk_index):
            logger.info(
                f"  [チャンク {chunk_index + 1}/{total_chunks}] "
                f"{offset_sec/60:.1f}分〜 → キャッシュ済みのためスキップ"
            )
            chunk_index += 1
            continue

        # 切り出す長さ: 通常は chunk_sec + overlap、最終チャンクは残り全部
        extract_duration = chunk_sec + overlap_sec

        logger.info(
            f"  [チャンク {chunk_index + 1}/{total_chunks}] "
            f"{offset_sec/60:.1f}分 〜 {(offset_sec + chunk_sec)/60:.1f}分 Groq API に送信中..."
        )

        # ── ffmpeg でチャンクを一時ファイルに切り出す（メモリを使わない）──
        # PHP の exec("ffmpeg -ss {$start} -t {$duration} -i input.mp3 output.mp3")
        # -ss: 開始時刻（秒）
        # -t : 切り出す長さ（秒）
        # -c copy: 再エンコードせずストリームコピー（高速・低負荷）
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(offset_sec),
                    "-t",  str(extract_duration),
                    "-i",  str(audio_path),
                    "-c",  "copy",          # 再エンコードなし（速い・メモリ低い）
                    "-loglevel", "error",   # エラー以外は非表示
                    str(tmp_path),
                ],
                check=True,
            )

            chunk_size_mb = tmp_path.stat().st_size / (1024 * 1024)
            logger.debug(f"    チャンクサイズ: {chunk_size_mb:.1f}MB")

            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    raw_segments = _call_groq_api_raw(client, tmp_path, cfg)
                    break  # 成功したらリトライループを抜ける
                except RateLimitError as e:
                    if attempt >= max_retries:
                        saved = cache.count_chunks(audio_id)
                        logger.error(
                            f"\n{'='*50}\n"
                            f"リトライ上限({max_retries}回)に達しました\n"
                            f"保存済み: {saved}/{total_chunks} チャンク\n"
                            f"{'='*50}"
                        )
                        raise
                    # エラーメッセージから待機秒数をパース
                    wait_sec = _parse_retry_after_sec(str(e)) + 15  # 15秒バッファ
                    logger.warning(
                        f"    レートリミット到達 → {wait_sec}秒後に自動リトライ "
                        f"({attempt + 1}/{max_retries}回目)"
                    )
                    for remaining in range(int(wait_sec), 0, -30):
                        logger.info(f"    待機中... 残り {remaining}秒")
                        time.sleep(min(30, remaining))
        finally:
            tmp_path.unlink(missing_ok=True)  # 一時ファイルを確実に削除

        # ── オーバーラップ除外 ＋ 絶対タイムスタンプに変換してDB保存 ──
        # Groqが返すタイムスタンプはチャンク内の相対時刻（0秒始まり）。
        # 例: チャンク2（10分〜）のセグメント start=5.0 → 絶対時刻 = 600 + 5.0 = 605秒
        #
        # 【オーバーラップ除外】
        # チャンク2以降は先頭 overlap_sec（10秒）が前チャンクと重複しているので捨てる。
        # 比較は相対時刻で行う（Groqの返却値がそうなっているため）。
        trim_start_rel = overlap_sec if chunk_index > 0 else 0.0
        filtered = [
            {
                "start": s["start"] + offset_sec,   # 相対 → 絶対時刻に変換
                "end":   s["end"]   + offset_sec,
                "text":  s["text"],
            }
            for s in raw_segments
            if s["start"] >= trim_start_rel          # 相対時刻で比較
        ]

        # ── DB に保存（タイムスタンプは絶対時刻）──
        cache.save_chunk(audio_id, chunk_index, offset_sec, filtered)
        logger.info(f"    → {len(filtered)}セグメント取得・保存完了")

        chunk_index += 1

    # ─── 全チャンクを DB から読み込んで Transcript に組み立て ───
    all_chunks = cache.load_all_chunks(audio_id)
    result_segments: list[TranscriptSegment] = []
    for chunk in all_chunks:
        for seg in chunk.segments:
            result_segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
            ))

    # start 順でソート（チャンク境界付近でずれる可能性があるため）
    result_segments.sort(key=lambda s: s.start)

    cache.close()

    transcript = Transcript(
        segments=result_segments,
        language=cfg.audio_language,
        audio_path=audio_path,
    )
    logger.success(
        f"Groq 文字起こし完了: {len(result_segments)}セグメント / "
        f"総時間: {transcript.duration_sec/60:.1f}分"
    )
    return transcript


def _parse_retry_after_sec(error_message: str) -> float:
    """
    Groq の RateLimitError メッセージから待機秒数をパースする。

    例: "Please try again in 4m36.5s."  → 276.5
        "Please try again in 30s."      → 30.0
        パース失敗時                    → 300.0（5分のデフォルト）
    """
    import re
    # "4m36.5s" 形式
    m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", error_message)
    if m:
        minutes = float(m.group(1) or 0)
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    return 300.0  # パース失敗時は5分待機


def _call_groq_api_raw(client, audio_path: Path, cfg: Config) -> list[dict]:
    """
    Groq API に音声ファイルを1つ送信し、セグメントを dict のリストで返す。
    キャッシュへの保存は呼び出し元（_transcribe_groq）が行う。

    返り値の形式:
        [{"start": 0.0, "end": 5.2, "text": "こんにちは"}, ...]

    【なぜ TranscriptSegment ではなく dict で返すか】
    DB（SQLite）に JSON 文字列として保存するため。
    dict → JSON → DB → JSON → dict → TranscriptSegment
    という流れで、DB との変換が自然になる。
    """
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, f),
            model=cfg.groq_whisper_model,
            language=cfg.audio_language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in (response.segments or [])
    ]
