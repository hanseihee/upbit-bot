"""SQLite 기반 거래 기록 및 포지션 관리."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import pandas as pd
from loguru import logger


class TradeDB:
    """비동기 SQLite 거래 데이터베이스."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """DB 초기화 및 테이블 생성."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_uuid TEXT UNIQUE NOT NULL,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                ord_type TEXT NOT NULL,
                price REAL,
                volume REAL,
                total_krw REAL,
                fee REAL DEFAULT 0,
                status TEXT DEFAULT 'wait',
                grid_level INTEGER,
                created_at TEXT NOT NULL,
                filled_at TEXT,
                related_order_uuid TEXT
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                entry_price REAL NOT NULL,
                volume REAL NOT NULL,
                grid_level INTEGER,
                stop_loss_price REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                pnl REAL,
                status TEXT DEFAULT 'open'
            );

            CREATE TABLE IF NOT EXISTS grid_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                grid_data TEXT NOT NULL,
                atr_at_creation REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market);

            CREATE TABLE IF NOT EXISTS momentum_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                position_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_momentum_market ON momentum_states(market);
            CREATE INDEX IF NOT EXISTS idx_momentum_active ON momentum_states(active);
        """)
        await self._db.commit()
        logger.info(f"데이터베이스 초기화 완료: {self._db_path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── 거래 기록 ───────────────────────────────────────

    async def save_trade(self, trade: dict[str, Any]) -> int:
        """거래 기록 저장."""
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT OR REPLACE INTO trades
               (order_uuid, market, side, ord_type, price, volume, total_krw,
                fee, status, grid_level, created_at, filled_at, related_order_uuid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade["order_uuid"],
                trade["market"],
                trade["side"],
                trade.get("ord_type", "limit"),
                trade.get("price"),
                trade.get("volume"),
                trade.get("total_krw"),
                trade.get("fee", 0),
                trade.get("status", "wait"),
                trade.get("grid_level"),
                trade.get("created_at", datetime.now().isoformat()),
                trade.get("filled_at"),
                trade.get("related_order_uuid"),
            ),
        )
        await self._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def update_trade_status(self, order_uuid: str, status: str, filled_at: str | None = None) -> None:
        assert self._db is not None
        if filled_at:
            await self._db.execute(
                "UPDATE trades SET status=?, filled_at=? WHERE order_uuid=?",
                (status, filled_at, order_uuid),
            )
        else:
            await self._db.execute(
                "UPDATE trades SET status=? WHERE order_uuid=?",
                (status, order_uuid),
            )
        await self._db.commit()

    # ── 포지션 관리 ─────────────────────────────────────

    async def save_position(self, position: dict[str, Any]) -> int:
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO positions
               (market, entry_price, volume, grid_level, stop_loss_price, entry_time, status)
               VALUES (?, ?, ?, ?, ?, ?, 'open')""",
            (
                position["market"],
                position["entry_price"],
                position["volume"],
                position.get("grid_level"),
                position.get("stop_loss_price"),
                position.get("entry_time", datetime.now().isoformat()),
            ),
        )
        await self._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def close_position(self, position_id: int, exit_price: float, pnl: float) -> None:
        assert self._db is not None
        await self._db.execute(
            """UPDATE positions
               SET status='closed', exit_time=?, exit_price=?, pnl=?
               WHERE id=?""",
            (datetime.now().isoformat(), exit_price, pnl, position_id),
        )
        await self._db.commit()

    async def get_open_positions(self, market: str | None = None) -> list[dict]:
        assert self._db is not None
        if market:
            cursor = await self._db.execute(
                "SELECT * FROM positions WHERE status='open' AND market=?", (market,)
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM positions WHERE status='open'"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_expired_positions(self, max_hours: int) -> list[dict]:
        """유지 시간 초과 포지션 조회."""
        assert self._db is not None
        cutoff = (datetime.now() - timedelta(hours=max_hours)).isoformat()
        cursor = await self._db.execute(
            "SELECT * FROM positions WHERE status='open' AND entry_time < ?",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── 손익 조회 ───────────────────────────────────────

    async def get_daily_pnl(self, date: str | None = None) -> float:
        """일일 실현 손익 합계."""
        assert self._db is not None
        target = date or datetime.now().strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM positions WHERE exit_time LIKE ? AND status='closed'",
            (f"{target}%",),
        )
        row = await cursor.fetchone()
        return float(row[0]) if row else 0.0

    async def get_trade_history(self, days: int = 7) -> pd.DataFrame:
        """최근 N일간 거래 내역."""
        assert self._db is not None
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = await self._db.execute(
            "SELECT * FROM trades WHERE created_at > ? ORDER BY created_at DESC",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ── 그리드 상태 ─────────────────────────────────────

    async def save_grid_state(self, market: str, grid_data: str, atr: float) -> int:
        assert self._db is not None
        now = datetime.now().isoformat()
        # 기존 그리드 비활성화
        await self._db.execute(
            "UPDATE grid_states SET active=0 WHERE market=? AND active=1", (market,)
        )
        cursor = await self._db.execute(
            """INSERT INTO grid_states (market, grid_data, atr_at_creation, created_at, updated_at, active)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (market, grid_data, atr, now, now),
        )
        await self._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_active_grid(self, market: str) -> dict | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM grid_states WHERE market=? AND active=1 ORDER BY id DESC LIMIT 1",
            (market,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ── 모멘텀 상태 ─────────────────────────────────────

    async def save_momentum_state(self, market: str, position_data: str) -> int:
        """모멘텀 포지션 상태 저장."""
        assert self._db is not None
        now = datetime.now().isoformat()
        await self._db.execute(
            "UPDATE momentum_states SET active=0 WHERE market=? AND active=1",
            (market,),
        )
        cursor = await self._db.execute(
            """INSERT INTO momentum_states
               (market, position_data, created_at, updated_at, active)
               VALUES (?, ?, ?, ?, 1)""",
            (market, position_data, now, now),
        )
        await self._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_active_momentum(self, market: str) -> dict | None:
        """활성 모멘텀 포지션 조회."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM momentum_states WHERE market=? AND active=1 "
            "ORDER BY id DESC LIMIT 1",
            (market,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def close_momentum_state(self, market: str) -> None:
        """모멘텀 포지션 비활성화."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE momentum_states SET active=0, updated_at=? WHERE market=? AND active=1",
            (datetime.now().isoformat(), market),
        )
        await self._db.commit()

    # ── 봇 상태 ─────────────────────────────────────────

    async def set_state(self, key: str, value: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, datetime.now().isoformat()),
        )
        await self._db.commit()

    async def get_state(self, key: str) -> str | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT value FROM bot_state WHERE key=?", (key,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None
