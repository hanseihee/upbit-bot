"""모멘텀/브레이크아웃 포지션 관리. 트레일링 스톱 기반."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd


class MomentumStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    EXITING = "exiting"
    CLOSED = "closed"


@dataclass
class MomentumPosition:
    """모멘텀 전략 포지션."""

    market: str
    entry_price: float
    volume: float
    order_krw: float
    entry_time: str
    status: MomentumStatus = MomentumStatus.PENDING
    order_uuid: str | None = None

    # 트레일링 스톱
    highest_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_pct: float = 0.02

    # ATR 기반 스톱
    atr_at_entry: float = 0.0
    atr_stop_multiplier: float = 1.5

    # 이탈 설정
    hard_stop_pct: float = 0.03
    rsi_exit: float = 75.0
    max_hold_minutes: int = 30

    # 청산 결과
    exit_price: float = 0.0
    exit_time: str = ""
    pnl_krw: float = 0.0
    pnl_pct: float = 0.0

    def update_trailing_stop(self, current_price: float, current_atr: float) -> float:
        """트레일링 스톱 갱신. 고점 갱신 시 스톱가 상향.

        퍼센트 스톱과 ATR 스톱 중 더 높은(타이트한) 값 사용.
        1% 이상 수익 시 스톱가가 진입가 아래로 내려가지 않음.
        """
        if current_price > self.highest_price:
            self.highest_price = current_price

        pct_stop = self.highest_price * (1 - self.trailing_stop_pct)

        if current_atr > 0:
            atr_stop = self.highest_price - (current_atr * self.atr_stop_multiplier)
            self.trailing_stop_price = max(pct_stop, atr_stop)
        else:
            self.trailing_stop_price = pct_stop

        # 1% 이상 수익 시 진입가 보전
        if self.highest_price > self.entry_price * 1.01:
            self.trailing_stop_price = max(
                self.trailing_stop_price, self.entry_price
            )

        return self.trailing_stop_price

    def should_exit(
        self,
        current_price: float,
        current_atr: float,
        rsi: float,
        macd_hist: float,
        macd_hist_prev: float,
    ) -> tuple[bool, str]:
        """이탈 조건 확인.

        1. 트레일링 스톱 도달
        2. RSI 과매수 소진
        3. MACD 히스토그램 양→음 전환
        4. 보유 시간 초과
        5. 하드 손절 (-3%)
        """
        self.update_trailing_stop(current_price, current_atr)

        # 1. 트레일링 스톱
        if self.trailing_stop_price > 0 and current_price <= self.trailing_stop_price:
            return True, (
                f"트레일링 스톱: {current_price:,.0f} <= {self.trailing_stop_price:,.0f}"
            )

        # 2. RSI 과매수
        if pd.notna(rsi) and rsi > self.rsi_exit:
            return True, f"RSI 과매수 소진: {rsi:.1f}"

        # 3. MACD 반전
        if (
            pd.notna(macd_hist)
            and pd.notna(macd_hist_prev)
            and macd_hist < 0
            and macd_hist_prev > 0
        ):
            return True, "MACD 히스토그램 음전환"

        # 4. 시간 초과
        try:
            entry_dt = datetime.fromisoformat(self.entry_time)
            elapsed_min = (datetime.now() - entry_dt).total_seconds() / 60
            if elapsed_min > self.max_hold_minutes:
                return True, f"보유시간 초과: {elapsed_min:.0f}분"
        except (ValueError, TypeError):
            pass

        # 5. 하드 손절
        if self.entry_price > 0 and current_price < self.entry_price * (1 - self.hard_stop_pct):
            return True, (
                f"하드 손절: {current_price:,.0f} < "
                f"{self.entry_price * (1 - self.hard_stop_pct):,.0f}"
            )

        return False, ""

    def calculate_pnl(self, exit_price: float) -> tuple[float, float]:
        """실현 PnL 계산 (KRW, %)."""
        if self.entry_price <= 0:
            return 0.0, 0.0
        pnl_pct = (exit_price - self.entry_price) / self.entry_price
        pnl_krw = self.order_krw * pnl_pct
        return pnl_krw, pnl_pct

    def to_json(self) -> str:
        return json.dumps(
            {
                "market": self.market,
                "entry_price": self.entry_price,
                "volume": self.volume,
                "order_krw": self.order_krw,
                "entry_time": self.entry_time,
                "status": self.status.value,
                "order_uuid": self.order_uuid,
                "highest_price": self.highest_price,
                "trailing_stop_price": self.trailing_stop_price,
                "trailing_stop_pct": self.trailing_stop_pct,
                "atr_at_entry": self.atr_at_entry,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "hard_stop_pct": self.hard_stop_pct,
                "rsi_exit": self.rsi_exit,
                "max_hold_minutes": self.max_hold_minutes,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, data: str) -> MomentumPosition:
        d = json.loads(data)
        return cls(
            market=d["market"],
            entry_price=d["entry_price"],
            volume=d["volume"],
            order_krw=d["order_krw"],
            entry_time=d["entry_time"],
            status=MomentumStatus(d["status"]),
            order_uuid=d.get("order_uuid"),
            highest_price=d.get("highest_price", d["entry_price"]),
            trailing_stop_price=d.get("trailing_stop_price", 0.0),
            trailing_stop_pct=d.get("trailing_stop_pct", 0.02),
            atr_at_entry=d.get("atr_at_entry", 0.0),
            atr_stop_multiplier=d.get("atr_stop_multiplier", 1.5),
            hard_stop_pct=d.get("hard_stop_pct", 0.03),
            rsi_exit=d.get("rsi_exit", 75.0),
            max_hold_minutes=d.get("max_hold_minutes", 30),
        )
