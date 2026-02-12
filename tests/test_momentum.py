"""모멘텀 포지션 관리 테스트."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from strategy.momentum import MomentumPosition, MomentumStatus


# ── 생성 테스트 ───────────────────────────────────────


def _make_position(**kwargs) -> MomentumPosition:
    defaults = {
        "market": "KRW-BTC",
        "entry_price": 90_000_000,
        "volume": 0.001,
        "order_krw": 90_000,
        "entry_time": datetime.now().isoformat(),
        "status": MomentumStatus.ACTIVE,
        "highest_price": 90_000_000,
        "trailing_stop_pct": 0.02,
        "atr_at_entry": 500_000,
        "atr_stop_multiplier": 1.5,
        "hard_stop_pct": 0.03,
        "rsi_exit": 75.0,
        "max_hold_minutes": 30,
    }
    defaults.update(kwargs)
    return MomentumPosition(**defaults)


class TestMomentumPositionCreation:
    def test_basic_creation(self):
        pos = _make_position()
        assert pos.market == "KRW-BTC"
        assert pos.entry_price == 90_000_000
        assert pos.status == MomentumStatus.ACTIVE

    def test_default_values(self):
        pos = MomentumPosition(
            market="KRW-ETH",
            entry_price=5_000_000,
            volume=0.1,
            order_krw=500_000,
            entry_time=datetime.now().isoformat(),
        )
        assert pos.status == MomentumStatus.PENDING
        assert pos.trailing_stop_pct == 0.02
        assert pos.hard_stop_pct == 0.03


# ── 트레일링 스톱 테스트 ─────────────────────────────


class TestTrailingStop:
    def test_initial_trailing_stop(self):
        pos = _make_position()
        stop = pos.update_trailing_stop(90_000_000, 500_000)
        # pct_stop = 90M * 0.98 = 88.2M
        # atr_stop = 90M - 500K * 1.5 = 89.25M
        assert stop == 89_250_000  # ATR 스톱이 더 높음

    def test_price_rise_updates_stop(self):
        pos = _make_position()
        pos.update_trailing_stop(90_000_000, 500_000)

        # 가격 상승
        stop = pos.update_trailing_stop(92_000_000, 500_000)
        assert pos.highest_price == 92_000_000
        # pct_stop = 92M * 0.98 = 90.16M
        # atr_stop = 92M - 750K = 91.25M
        assert stop == 91_250_000

    def test_price_drop_does_not_lower_stop(self):
        pos = _make_position()
        pos.update_trailing_stop(92_000_000, 500_000)
        stop_high = pos.trailing_stop_price

        # 가격 하락해도 스톱은 내려가지 않음
        stop = pos.update_trailing_stop(91_000_000, 500_000)
        assert stop >= stop_high
        assert pos.highest_price == 92_000_000  # 고점 유지

    def test_profit_protection(self):
        """1% 이상 수익 시 진입가 아래로 스톱이 내려가지 않음."""
        pos = _make_position(entry_price=90_000_000, highest_price=90_000_000)

        # 1% 이상 수익
        pos.update_trailing_stop(92_000_000, 500_000)
        assert pos.trailing_stop_price >= 90_000_000

    def test_zero_atr_uses_pct_only(self):
        pos = _make_position()
        stop = pos.update_trailing_stop(90_000_000, 0)
        expected = 90_000_000 * 0.98  # 88.2M
        assert stop == expected


# ── 이탈 조건 테스트 ─────────────────────────────────


class TestExitConditions:
    def test_trailing_stop_exit(self):
        pos = _make_position()
        pos.update_trailing_stop(92_000_000, 500_000)

        # 스톱 아래로 하락
        should, reason = pos.should_exit(88_000_000, 500_000, 60, 100, 200)
        assert should is True
        assert "트레일링 스톱" in reason

    def test_rsi_overbought_exit(self):
        pos = _make_position()
        pos.update_trailing_stop(90_000_000, 500_000)

        should, reason = pos.should_exit(95_000_000, 500_000, 78, 100, 200)
        assert should is True
        assert "RSI" in reason

    def test_macd_reversal_exit(self):
        pos = _make_position()
        pos.update_trailing_stop(90_000_000, 500_000)

        # MACD 양→음 전환
        should, reason = pos.should_exit(91_000_000, 500_000, 60, -50, 100)
        assert should is True
        assert "MACD" in reason

    def test_time_based_exit(self):
        past = (datetime.now() - timedelta(minutes=35)).isoformat()
        pos = _make_position(entry_time=past, max_hold_minutes=30)
        pos.update_trailing_stop(90_000_000, 500_000)

        should, reason = pos.should_exit(91_000_000, 500_000, 60, 100, 50)
        assert should is True
        assert "보유시간" in reason

    def test_hard_stop_exit(self):
        """트레일링 스톱보다 하드 손절이 먼저 걸리는 케이스."""
        pos = _make_position(
            entry_price=90_000_000,
            hard_stop_pct=0.03,
            trailing_stop_pct=0.10,  # 트레일링 스톱 10%로 넓게
            atr_stop_multiplier=0.0,  # ATR 스톱 비활성화
        )
        pos.update_trailing_stop(90_000_000, 0)  # ATR 0

        # 진입가 대비 -4% 하락 (트레일링 10%보다 작고, 하드 3%보다 큼)
        price = 90_000_000 * 0.96
        should, reason = pos.should_exit(price, 0, 40, 100, 200)
        assert should is True
        assert "하드 손절" in reason

    def test_no_exit_in_normal_conditions(self):
        pos = _make_position()
        pos.update_trailing_stop(90_000_000, 500_000)

        # 정상 범위: 가격 상승, RSI 정상, MACD 양수 유지
        should, reason = pos.should_exit(91_000_000, 500_000, 60, 150, 100)
        assert should is False
        assert reason == ""

    def test_macd_nan_no_exit(self):
        """MACD NaN일 때 MACD 조건으로 이탈하지 않음."""
        pos = _make_position()
        pos.update_trailing_stop(90_000_000, 500_000)

        should, _ = pos.should_exit(91_000_000, 500_000, 60, float("nan"), float("nan"))
        assert should is False


# ── PnL 계산 테스트 ──────────────────────────────────


class TestPnlCalculation:
    def test_profit(self):
        pos = _make_position(entry_price=90_000_000, order_krw=90_000)
        krw, pct = pos.calculate_pnl(91_800_000)  # +2%
        assert pct == pytest.approx(0.02, abs=1e-6)
        assert krw == pytest.approx(1_800, abs=1)

    def test_loss(self):
        pos = _make_position(entry_price=90_000_000, order_krw=90_000)
        krw, pct = pos.calculate_pnl(87_300_000)  # -3%
        assert pct == pytest.approx(-0.03, abs=1e-6)
        assert krw == pytest.approx(-2_700, abs=1)

    def test_zero_entry_price(self):
        pos = _make_position(entry_price=0)
        krw, pct = pos.calculate_pnl(90_000_000)
        assert krw == 0.0
        assert pct == 0.0


# ── 직렬화 테스트 ────────────────────────────────────


class TestSerialization:
    def test_to_json_and_back(self):
        pos = _make_position(
            entry_price=90_000_000,
            volume=0.001,
            order_krw=90_000,
            highest_price=92_000_000,
            trailing_stop_price=91_000_000,
        )

        json_str = pos.to_json()
        restored = MomentumPosition.from_json(json_str)

        assert restored.market == pos.market
        assert restored.entry_price == pos.entry_price
        assert restored.volume == pos.volume
        assert restored.order_krw == pos.order_krw
        assert restored.status == pos.status
        assert restored.highest_price == pos.highest_price
        assert restored.trailing_stop_pct == pos.trailing_stop_pct
        assert restored.atr_stop_multiplier == pos.atr_stop_multiplier
        assert restored.hard_stop_pct == pos.hard_stop_pct
        assert restored.rsi_exit == pos.rsi_exit

    def test_json_is_valid(self):
        pos = _make_position()
        data = json.loads(pos.to_json())
        assert data["market"] == "KRW-BTC"
        assert data["entry_price"] == 90_000_000


# ── Status enum 테스트 ───────────────────────────────


class TestMomentumStatus:
    def test_status_values(self):
        assert MomentumStatus.PENDING == "pending"
        assert MomentumStatus.ACTIVE == "active"
        assert MomentumStatus.EXITING == "exiting"
        assert MomentumStatus.CLOSED == "closed"
