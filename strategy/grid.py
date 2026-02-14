"""Adaptive Grid Trading 로직. ATR 기반 동적 그리드 간격."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum


class GridSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class GridStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"       # 주문 등록됨
    FILLED = "filled"       # 체결됨
    CANCELLED = "cancelled"


@dataclass
class GridLevel:
    level: int
    price: float
    side: GridSide
    status: GridStatus = GridStatus.PENDING
    order_uuid: str | None = None
    volume: float = 0.0

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "price": self.price,
            "side": self.side.value,
            "status": self.status.value,
            "order_uuid": self.order_uuid,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GridLevel:
        return cls(
            level=d["level"],
            price=d["price"],
            side=GridSide(d["side"]),
            status=GridStatus(d["status"]),
            order_uuid=d.get("order_uuid"),
            volume=d.get("volume", 0.0),
        )


@dataclass
class Grid:
    market: str
    levels: list[GridLevel] = field(default_factory=list)
    base_price: float = 0.0
    atr_at_creation: float = 0.0
    grid_spacing: float = 0.0
    stop_loss_price: float = 0.0

    def to_json(self) -> str:
        return json.dumps({
            "market": self.market,
            "base_price": self.base_price,
            "atr_at_creation": self.atr_at_creation,
            "grid_spacing": self.grid_spacing,
            "stop_loss_price": self.stop_loss_price,
            "levels": [l.to_dict() for l in self.levels],
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> Grid:
        d = json.loads(data)
        grid = cls(
            market=d["market"],
            base_price=d["base_price"],
            atr_at_creation=d["atr_at_creation"],
            grid_spacing=d["grid_spacing"],
            stop_loss_price=d["stop_loss_price"],
        )
        grid.levels = [GridLevel.from_dict(l) for l in d["levels"]]
        return grid

    @property
    def buy_levels(self) -> list[GridLevel]:
        return [l for l in self.levels if l.side == GridSide.BUY]

    @property
    def sell_levels(self) -> list[GridLevel]:
        return [l for l in self.levels if l.side == GridSide.SELL]

    @property
    def pending_buys(self) -> list[GridLevel]:
        return [l for l in self.buy_levels if l.status == GridStatus.PENDING]

    @property
    def active_orders(self) -> list[GridLevel]:
        return [l for l in self.levels if l.status == GridStatus.ACTIVE]


class AdaptiveGrid:
    """ATR 기반 동적 그리드 생성기."""

    def __init__(
        self,
        atr_multiplier: float = 0.5,
        stop_loss_pct: float = 0.05,
        fee_rate: float = 0.0005,
        sell_target_mult: float = 1.0,
    ) -> None:
        self._atr_mult = atr_multiplier
        self._stop_loss_pct = stop_loss_pct
        self._fee_rate = fee_rate
        self._sell_target_mult = sell_target_mult

    def calculate_grid(
        self,
        market: str,
        current_price: float,
        atr: float,
        levels: int = 5,
    ) -> Grid:
        """ATR 기반 동적 그리드 생성.

        매수 그리드: 현재가 아래로 levels개
        매도 그리드: 각 매수 레벨 +1 위에서 매도

        예시 (현재가 90,000,000, ATR 2,000,000, mult 0.5):
        spacing = 2,000,000 * 0.5 = 1,000,000
        Grid 5: 91,000,000 (매도)
        Grid 4: 90,000,000 (매도) ← 현재가
        Grid 3: 89,000,000 (매수)
        Grid 2: 88,000,000 (매수)
        Grid 1: 87,000,000 (매수)
        """
        spacing = atr * self._atr_mult

        # 최소 간격: 수수료 × 2 이상이어야 수익
        min_spacing = current_price * self._fee_rate * 3  # 수수료 3배 이상
        spacing = max(spacing, min_spacing)

        # Upbit 호가 단위에 맞게 반올림
        spacing = self._round_to_tick(spacing, current_price)

        grid = Grid(
            market=market,
            base_price=current_price,
            atr_at_creation=atr,
            grid_spacing=spacing,
        )

        # 매수 그리드 (현재가 아래)
        for i in range(levels):
            buy_price = self._round_to_tick(
                current_price - spacing * (i + 1), current_price
            )
            grid.levels.append(GridLevel(
                level=i + 1,
                price=buy_price,
                side=GridSide.BUY,
            ))

        # 매도 그리드 (각 매수 레벨 + sell_target_mult * spacing 위)
        for i in range(levels):
            buy_price_i = current_price - spacing * (i + 1)
            sell_price = self._round_to_tick(
                buy_price_i + spacing * self._sell_target_mult, current_price
            )
            grid.levels.append(GridLevel(
                level=i + 1,
                price=sell_price,
                side=GridSide.SELL,
            ))

        # 손절가 설정 (최하단 매수가 - stop_loss_pct)
        lowest_buy = min(l.price for l in grid.buy_levels)
        grid.stop_loss_price = self._round_to_tick(
            lowest_buy * (1 - self._stop_loss_pct), current_price
        )

        return grid

    def should_update_grid(self, current_atr: float, grid_atr: float, threshold: float = 0.2) -> bool:
        """ATR 변화가 threshold(20%) 이상이면 그리드 재설정 필요."""
        if grid_atr <= 0:
            return True
        change = abs(current_atr - grid_atr) / grid_atr
        return change >= threshold

    @staticmethod
    def _round_to_tick(price: float, reference_price: float) -> float:
        """Upbit 호가 단위에 맞게 반올림.

        KRW 마켓 호가 단위:
        >= 2,000,000: 1,000원
        >= 1,000,000: 500원
        >= 500,000: 100원
        >= 100,000: 50원
        >= 10,000: 10원
        >= 1,000: 5원
        >= 100: 1원
        >= 10: 0.1원
        >= 1: 0.01원
        < 1: 0.001원
        """
        p = reference_price
        if p >= 2_000_000:
            tick = 1000
        elif p >= 1_000_000:
            tick = 500
        elif p >= 500_000:
            tick = 100
        elif p >= 100_000:
            tick = 50
        elif p >= 10_000:
            tick = 10
        elif p >= 1_000:
            tick = 5
        elif p >= 100:
            tick = 1
        elif p >= 10:
            tick = 0.1
        elif p >= 1:
            tick = 0.01
        else:
            tick = 0.001

        return round(round(price / tick) * tick, 8)
