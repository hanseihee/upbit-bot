"""리스크 관리자 테스트."""

import pytest
from datetime import datetime, timedelta

from config.settings import TradingConfig
from trading.risk_manager import RiskManager


class TestPositionSizing:
    def setup_method(self):
        self.config = TradingConfig()
        self.risk = RiskManager(self.config)

    def test_basic_sizing(self):
        """기본 포지션 사이징 계산."""
        # 잔고 300,000 * 신뢰도 0.8 * 코인당 35% / 5레벨 = 16,800
        size = self.risk.calculate_position_size(300_000, 90_000_000, 0.8, 5)
        assert size == 16_800

    def test_low_confidence_returns_zero(self):
        """신뢰도가 기준 미달이면 0 반환."""
        size = self.risk.calculate_position_size(300_000, 90_000_000, 0.3, 5)
        assert size == 0

    def test_below_min_order_returns_zero(self):
        """최소주문금액 미달 시 0 반환."""
        # 잔고 10,000 * 0.8 * 0.5 / 5 = 400 < 5,000
        size = self.risk.calculate_position_size(10_000, 90_000_000, 0.8, 5)
        assert size == 0


class TestDrawdown:
    def setup_method(self):
        self.config = TradingConfig(max_drawdown_pct=0.10)
        self.risk = RiskManager(self.config)
        self.risk.set_initial_balance(500_000)

    def test_no_drawdown(self):
        """드로다운 없으면 정상."""
        check = self.risk.check_drawdown(500_000)
        assert not check.should_stop
        assert not check.should_close_all

    def test_within_limit(self):
        """한도 내 드로다운."""
        check = self.risk.check_drawdown(460_000)  # -8%
        assert not check.should_stop

    def test_exceeds_limit(self):
        """한도 초과 드로다운."""
        check = self.risk.check_drawdown(440_000)  # -12%
        assert check.should_stop
        assert check.should_close_all

    def test_exact_limit(self):
        """정확히 한도."""
        check = self.risk.check_drawdown(450_000)  # -10%
        assert check.should_stop


class TestStopLoss:
    def setup_method(self):
        self.config = TradingConfig(stop_loss_pct=0.05)
        self.risk = RiskManager(self.config)

    def test_no_stop_loss(self):
        """가격 상승 시 손절 아님."""
        assert not self.risk.check_stop_loss(100_000, 105_000)

    def test_small_loss(self):
        """소폭 하락 시 손절 아님."""
        assert not self.risk.check_stop_loss(100_000, 97_000)  # -3%

    def test_stop_loss_triggered(self):
        """5% 이상 하락 시 손절."""
        assert self.risk.check_stop_loss(100_000, 94_000)  # -6%


class TestPositionAge:
    def setup_method(self):
        self.config = TradingConfig(max_position_age_hours=48)
        self.risk = RiskManager(self.config)

    def test_fresh_position(self):
        """신규 포지션."""
        now = datetime.now().isoformat()
        assert not self.risk.check_position_age(now)

    def test_expired_position(self):
        """48시간 초과 포지션."""
        old = (datetime.now() - timedelta(hours=50)).isoformat()
        assert self.risk.check_position_age(old)


class TestOrderValidation:
    def setup_method(self):
        self.config = TradingConfig(min_order_krw=5_000)
        self.risk = RiskManager(self.config)

    def test_valid_order(self):
        valid, _ = self.risk.validate_order(10_000, 100_000)
        assert valid

    def test_below_minimum(self):
        valid, reason = self.risk.validate_order(3_000, 100_000)
        assert not valid
        assert "최소" in reason

    def test_insufficient_balance(self):
        valid, reason = self.risk.validate_order(10_000, 5_000)
        assert not valid
        assert "잔고" in reason
