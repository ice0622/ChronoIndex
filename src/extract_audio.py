"""
src/extract_audio.py
────────────────────────────────────────────────────────────
YouTube URLから音声ファイルをダウンロードするモジュール

【Python設計思想 ①: 単一責任の原則】
このファイルは「音声を取得する」だけを責務とする。
ASR（文字起こし）や境界検出は別ファイルに書く。
→ 1ファイル1責務にすることで、テストしやすく、壊れても影響範囲が小さい。

【Python設計思想 ②: dataclass で返り値を型安全にする】
関数が複数の値を返すとき dict を使うと間違えやすい。
dataclass を使うとフィールド名が補完され、型チェックも効く。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path  # ← os.path より直感的なパス操作。Pythonらしい書き方。

from loguru import logger


# ─────────────────────────────────────────
# 返り値の型定義
# ─────────────────────────────────────────
@dataclass
class AudioInfo:
    """
    ダウンロード結果を格納するデータクラス。

    【dataclass とは】
    @dataclass をつけるだけで __init__ / __repr__ / __eq__ が自動生成される。
    Pythonでは「ボイラープレートを書かない」ことが美徳とされており、
    dataclass はその典型例。
    """
    path: Path        # 保存先のファイルパス
    title: str        # 動画タイトル
    duration_sec: int # 動画の長さ（秒）
    url: str          # 元のURL


# ─────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────
def extract_audio(url: str, output_dir: Path) -> AudioInfo:
    """
    YouTube URLから音声（mp3）をダウンロードして保存する。

    Args:
        url:        YouTube動画のURL
        output_dir: 保存先ディレクトリ

    Returns:
        AudioInfo: ダウンロードされたファイルの情報

    Raises:
        RuntimeError: ダウンロードに失敗した場合

    【Python設計思想 ③: 型ヒント（Type Hints）】
    引数と返り値に型を書く。実行時には強制されないが、
    - エディタの補完が効く
    - コードを読む人への「仕様書」になる
    - mypyなどで静的チェックができる
    Python 3.5+ から標準機能。
    """

    # Path オブジェクトは / 演算子でパスを繋げられる（Pythonらしい書き方）
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"音声ダウンロード開始: {url}")

    # ─── Step A: 動画メタ情報を取得（実際のダウンロードはしない）───
    meta = _fetch_metadata(url)
    logger.info(f"タイトル: {meta['title']} / 長さ: {meta['duration']}秒")

    # ─── Step B: 音声をダウンロード ───
    # ファイル名から使えない文字を除いて保存する
    safe_title = _sanitize_filename(meta["title"])
    output_path = output_dir / f"{safe_title}.mp3"

    if output_path.exists():
        # 【設計判断】同じファイルがあれば再ダウンロードしない（冪等性）
        logger.info(f"キャッシュ済みのため再利用: {output_path}")
    else:
        _download_audio(url, output_path)

    return AudioInfo(
        path=output_path,
        title=meta["title"],
        duration_sec=int(meta["duration"]),
        url=url,
    )


# ─────────────────────────────────────────
# 内部ヘルパー関数（プライベート扱い）
# ─────────────────────────────────────────
def _fetch_metadata(url: str) -> dict:
    """
    動画のメタ情報（タイトル・長さ）だけを取得する。

    【Python設計思想 ④: アンダースコアで「内部関数」を示す】
    _ で始まる関数は「このモジュール内だけで使う」という慣習的なシグナル。
    Java の private のような強制力はないが、チームの読みやすさが上がる。
    """
    cmd = [
        "yt-dlp",
        "--print", "%(title)s\t%(duration)s",  # タブ区切りで title と duration を出力
        "--no-download",
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 【学習ポイント】check=True を使わず戻り値を自分でチェックする
    # check=True だと CalledProcessError が飛び、stderr（本当のエラー原因）が
    # 表示されないまま例外になる。
    # yt-dlp のように「エラーメッセージが stderr に出るツール」では
    # 自分でハンドリングした方がデバッグしやすい。
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp --print が失敗しました (exit={result.returncode})\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr}"
        )

    parts = result.stdout.strip().split("\t")
    if len(parts) != 2:
        raise RuntimeError(f"メタ情報の取得に失敗しました: {result.stdout!r}")

    return {"title": parts[0], "duration": parts[1]}


def _download_audio(url: str, output_path: Path) -> None:
    """
    yt-dlp で音声をmp3に変換してダウンロードする。

    【None 返り値について】
    「副作用だけを起こす関数」は明示的に -> None と書く。
    呼び出し元に「返り値を使う必要はない」と伝えられる。
    """
    cmd = [
        "yt-dlp",
        "--extract-audio",           # 映像なし、音声のみ
        "--audio-format", "mp3",     # mp3に変換（Whisperが読める形式）
        "--audio-quality", "0",      # 最高品質（ファイルサイズより精度優先）
        "--no-playlist",             # 再生リストは処理しない
        "--output", str(output_path.with_suffix("")),  # 拡張子はyt-dlpが付ける
        url,
    ]
    logger.debug(f"実行コマンド: {' '.join(cmd)}")

    # 【capture_output=False（デフォルト）にする理由】
    # capture_output=True にすると stdout/stderr が全て飲み込まれ、
    # yt-dlp の進捗バーが見えなくなる（フリーズしているように見える）。
    # ダウンロードのような長時間処理では進捗を見せることが重要。
    # エラー検出は returncode だけで十分。
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp が失敗しました (exit={result.returncode})\n"
            f"上記の出力を確認してください"
        )

    logger.success(f"ダウンロード完了: {output_path}")


def _sanitize_filename(name: str) -> str:
    """
    ファイル名に使えない文字を除去する。

    【リスト内包表記 - Pythonの強み】
    Pythonの最も「らしい」文法の一つ。
    [式 for 変数 in イテラブル if 条件] という形で1行でリストを作れる。

    他言語での書き方:
        result = []
        for c in name:
            if c not in invalid_chars:
                result.append(c)

    Pythonらしい書き方（内包表記）:
        result = [c for c in name if c not in invalid_chars]
    """
    invalid_chars = set(r'\/:*?"<>|')
    sanitized = "".join(c for c in name if c not in invalid_chars)
    # 【generator expression】
    # [...] ではなく (...) を使うと「generatorオブジェクト」になる。
    # join のように「一度しか読まない」用途ではリストを作らずメモリ効率が良い。
    return sanitized[:100]  # ファイル名が長すぎる場合に切り詰める
