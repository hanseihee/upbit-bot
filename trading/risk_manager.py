"""리스크 관리 모듈. 손절/포지션사이징/드로다운 관리."""

from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import dataclass

from loguru import logger

from config.settings import TradingConfig
from utils.notifier import Notifier


@dataclass
class RiskCheck:
    """리스크 체크 결과."""
    should_stop: bool = False
    should_close_all: bool = False
    positions_to_close: list[int] | None = None
    reason: str = ""


class RiskManager:
    """리스크 관리자. 포지션사이징, 손절, 드로다운 관리."""

    def __init__(self, config: TradingConfig, notifier: Notifier | None = None) -> None:
        self._config = config
        self._notifier = notifier or Notifier()
        self._initial_balance: float | None = None

    def set_initial_balance(self, balance: float) -> None:
        """초기 잔고 설정 (봇 시작 시 1회)."""
        self._initial_balance = balance
        logger.info(f"초기 잔고 설정: {balance:,.0f} KRW")

    def calculate_position_size(
        self,
        available_krw: float,
        price: float,
        confidence: float,
        num_grid_levels: int,
    ) -> float:
        """그리드 레벨당 주문 금액 계산 (KRW).

        공식: 주문당 = 가용잔고 * 신뢰도 * 코인당비율 / 그리드수
        예: 300,000 * 0.8 * 0.5 / 5 = 24,000원
        """
        if confidence < self._config.min_signal_confidence:
            return 0.0

        per_level = (
            available_krw
            * confidence
            * self._config.max_per_coin_ratio
            / max(num_grid_levels, 1)
        )

        # 최소 주문금액 체크
        if per_level < self._config.min_order_krw:
            logger.debug(
                f"주문금액 {per_level:,.0f}원 < 최소 {self._config.min_order_krw:,.0f}원"
            )
            return 0.0

        return round(per_level, 0)

    def calculate_volume(self, order_krw: float, price: float) -> float:
        """KRW 주문금액을 코인 수량으로 변환."""
        if price <= 0:
            return 0.0
        return order_krw / price

    def check_drawdown(self, current_total: float) -> RiskCheck:
        """총자산 드로다운 체크. -10% 도달 시 전체 청산 트리거."""
        if self._initial_balance is None:
            return RiskCheck()

        if self._initial_balance <= 0:
            return RiskCheck()

        drawdown = (self._initial_balance - current_total) / self._initial_balance

        if drawdown >= self._config.max_drawdown_pct:
            self._notifier.drawdown_alert(drawdown, self._config.max_drawdown_pct)
            return RiskCheck(
                should_stop=True,
                should_close_all=True,
                reason=f"드로다운 한도 초과: {drawdown:.2%} >= {self._config.max_drawdown_pct:.2%}",
            )

        # 경고 (한도의 70% 이상)
        if drawdown >= self._config.max_drawdown_pct * 0.7:
            logger.warning(f"드로다운 경고: {drawdown:.2%}")

        return RiskCheck()

    def check_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """개별 포지션 손절가 도달 여부."""
        if entry_price <= 0:
            return False
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self._config.stop_loss_pct

    def check_position_age(self, entry_time: str) -> bool:
        """포지션 유지 시간 초과 여부."""
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            max_age = timedelta(hours=self._config.max_position_age_hours)
            return datetime.now() - entry_dt > max_age
        except (ValueError, TypeError):
            return False

    def should_take_profit(self, entry_price: float, current_price: float, grid_spacing: float) -> bool:
        """그리드 1단계 이상 수익 시 이익 실현 여부."""
        if entry_price <= 0 or grid_spacing <= 0:
            return False
        profit = current_price - entry_price
        return profit >= grid_spacing

    def calculate_momentum_size(
        self,
        available_krw: float,
        confidence: float,
    ) -> float:
        """모멘텀 포지션 주문 금액 계산 (KRW).

        공식: 가용잔고 × 신뢰도 × momentum_position_ratio
        """
        if confidence < self._config.min_signal_confidence:
            return 0.0

        order_krw = (
            available_krw
            * confidence
            * self._config.momentum_position_ratio
        )
        order_krw = round(order_krw, 0)

        if order_krw < self._config.min_order_krw:
            return 0.0

        return order_krw

    def validate_order(self, order_krw: float, available_krw: float) -> tuple[bool, str]:
        """주문 유효성 검증."""
        if order_krw < self._config.min_order_krw:
            return False, f"최소 주문금액 미달: {order_krw:,.0f} < {self._config.min_order_krw:,.0f}"

        if order_krw > available_krw:
            return False, f"잔고 부족: 주문 {order_krw:,.0f} > 가용 {available_krw:,.0f}"

        return True, ""
