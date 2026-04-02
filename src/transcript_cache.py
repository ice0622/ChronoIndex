"""
src/transcript_cache.py
────────────────────────────────────────────────────────────
文字起こしチャンクを SQLite にキャッシュするモジュール

【PHPer 向け解説】
PHP + MySQL では PDO でこんな風に書きますが、
Python + SQLite では sqlite3 という標準ライブラリを使います。
インストール不要・pip も不要・Python に最初から入っています。

MySQLとの主な違い:
  - SQLite はサーバーなしで動く（ファイル1つがDB全体）
  - 型は TEXT / INTEGER / REAL / BLOB の4種類だけ
  - 構文はほぼ同じ（CREATE TABLE, INSERT, SELECT, PRIMARY KEY など）

【キャッシュが必要な理由】
Groq API には 7,200秒/時のレート制限があります。
2時間動画を 10分チャンクで処理すると12チャンク必要で、
制限に達した時点で中断されます。

キャッシュがないと: チャンク6で失敗 → チャンク1〜5の処理がすべて無駄
キャッシュがあると: チャンク6で失敗 → 再実行時にチャンク1〜5はスキップ
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger


# ─────────────────────────────────────────
# DB スキーマ（MySQL と同じ DDL 構文）
# ─────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transcript_chunks (
    audio_id      TEXT    NOT NULL,   -- 音声ファイルの MD5 ハッシュ（ファイルの指紋）
    chunk_index   INTEGER NOT NULL,   -- 何番目のチャンクか（0始まり）
    offset_sec    REAL    NOT NULL,   -- 元動画の何秒目から切り出したか
    segments_json TEXT    NOT NULL,   -- 文字起こし結果（JSON文字列）
    created_at    TEXT    NOT NULL,   -- 処理日時
    PRIMARY KEY (audio_id, chunk_index)
)
"""


@dataclass
class CachedChunk:
    """キャッシュから取り出した1チャンク分のデータ"""
    chunk_index: int
    offset_sec:  float
    segments:    list[dict]   # [{"start": 0.0, "end": 5.0, "text": "..."}, ...]


class TranscriptCache:
    """
    SQLite を使った文字起こしチャンクのキャッシュ。

    【PHP の PDO との対応】
    PHP:    $pdo = new PDO('sqlite:/path/to/db.sqlite');
    Python: conn = sqlite3.connect('/path/to/db.sqlite')

    PHP:    $stmt = $pdo->prepare("SELECT ...");
            $stmt->execute([$param]);
            $row = $stmt->fetch();
    Python: cur = conn.execute("SELECT ...", (param,))
            row = cur.fetchone()
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Docker コンテナ内で安全に動作させるオプション
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row  # カラム名でアクセスできるように（PHP の fetch(PDO::FETCH_ASSOC) 相当）
        self._ensure_table()
        logger.debug(f"TranscriptCache 初期化: {db_path}")

    def _ensure_table(self) -> None:
        """テーブルがなければ作る（MySQL の CREATE TABLE IF NOT EXISTS と同じ）"""
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    # ─────────────────────────────────────────
    # audio_id の生成
    # ─────────────────────────────────────────
    @staticmethod
    def make_audio_id(audio_path: Path) -> str:
        """
        音声ファイルの "指紋" を作る。
        ファイルパス + ファイルサイズの MD5 ハッシュを使う。

        【なぜファイル全体の MD5 を使わないか】
        ファイルが 250MB もあると MD5 計算だけで数秒かかる。
        パス + サイズの組み合わせは実用上十分ユニーク。

        PHP でいうと:
            $audioId = md5($filePath . filesize($filePath));
        """
        size  = audio_path.stat().st_size
        raw   = f"{audio_path.resolve()}:{size}"
        return hashlib.md5(raw.encode()).hexdigest()  # 例: "a3f2c1..."（32文字）

    # ─────────────────────────────────────────
    # チャンクの存在確認（SELECT）
    # ─────────────────────────────────────────
    def has_chunk(self, audio_id: str, chunk_index: int) -> bool:
        """
        指定チャンクがキャッシュ済みかどうか確認する。

        PHP でいうと:
            $row = $pdo->query(
                "SELECT 1 FROM transcript_chunks
                  WHERE audio_id=? AND chunk_index=?",
                [$audioId, $chunkIndex]
            )->fetch();
            return (bool)$row;
        """
        cur = self._conn.execute(
            "SELECT 1 FROM transcript_chunks WHERE audio_id=? AND chunk_index=?",
            (audio_id, chunk_index),
        )
        return cur.fetchone() is not None

    # ─────────────────────────────────────────
    # チャンクの保存（INSERT OR REPLACE）
    # ─────────────────────────────────────────
    def save_chunk(
        self,
        audio_id:    str,
        chunk_index: int,
        offset_sec:  float,
        segments:    list[dict],
    ) -> None:
        """
        チャンクの文字起こし結果を保存する。
        既に同じ (audio_id, chunk_index) があれば上書き。

        PHP でいうと:
            INSERT INTO transcript_chunks (...) VALUES (...)
            ON DUPLICATE KEY UPDATE ...
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO transcript_chunks
                (audio_id, chunk_index, offset_sec, segments_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                audio_id,
                chunk_index,
                offset_sec,
                json.dumps(segments, ensure_ascii=False),  # dict → JSON文字列
                datetime.utcnow().isoformat(),
            ),
        )
        self._conn.commit()

    # ─────────────────────────────────────────
    # チャンクの読み込み（SELECT）
    # ─────────────────────────────────────────
    def load_chunk(self, audio_id: str, chunk_index: int) -> CachedChunk | None:
        """
        キャッシュからチャンクを取り出す。
        なければ None を返す。

        PHP でいうと:
            $row = $pdo->query("SELECT ...", [$audioId, $chunkIndex])->fetch();
        """
        cur = self._conn.execute(
            "SELECT chunk_index, offset_sec, segments_json "
            "FROM transcript_chunks WHERE audio_id=? AND chunk_index=?",
            (audio_id, chunk_index),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return CachedChunk(
            chunk_index=row["chunk_index"],
            offset_sec=row["offset_sec"],
            segments=json.loads(row["segments_json"]),  # JSON文字列 → dict のリスト
        )

    # ─────────────────────────────────────────
    # 処理済みチャンク数の確認
    # ─────────────────────────────────────────
    def count_chunks(self, audio_id: str) -> int:
        """この動画で保存済みのチャンク数を返す"""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM transcript_chunks WHERE audio_id=?",
            (audio_id,),
        )
        return cur.fetchone()[0]

    # ─────────────────────────────────────────
    # 全チャンクをまとめて取得
    # ─────────────────────────────────────────
    def load_all_chunks(self, audio_id: str) -> list[CachedChunk]:
        """
        この動画の全チャンクを chunk_index 順で返す。
        全チャンク完了後に Transcript に組み立てるために使う。
        """
        cur = self._conn.execute(
            "SELECT chunk_index, offset_sec, segments_json "
            "FROM transcript_chunks WHERE audio_id=? ORDER BY chunk_index ASC",
            (audio_id,),
        )
        return [
            CachedChunk(
                chunk_index=row["chunk_index"],
                offset_sec=row["offset_sec"],
                segments=json.loads(row["segments_json"]),
            )
            for row in cur.fetchall()
        ]

    def close(self) -> None:
        """DB 接続を閉じる（PHP の $pdo = null; 相当）"""
        self._conn.close()
