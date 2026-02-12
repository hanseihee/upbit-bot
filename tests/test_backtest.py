"""백테스트 엔진 테스트."""

import numpy as np
import pandas as pd
import pytest

from config.settings import TradingConfig
from backtest.engine import BacktestEngine


def _make_candle_df(n: int = 500, base_price: float = 100_000, volatility: float = 0.02) -> pd.DataFrame:
    """백테스트용 합성 캔들 데이터 생성.

    랜덤워크 + 평균 회귀 특성을 가진 가격 데이터.
    """
    np.random.seed(42)
    prices = [base_price]
    for _ in range(n - 1):
        # 평균 회귀: base_price 근처로 복귀하려는 경향
        mean_revert = (base_price - prices[-1]) * 0.01
        random_walk = np.random.normal(0, base_price * volatility)
        new_price = prices[-1] + mean_revert + random_walk
        prices.append(max(new_price, base_price * 0.5))

    highs = [p * (1 + np.random.uniform(0.001, 0.02)) for p in prices]
    lows = [p * (1 - np.random.uniform(0.001, 0.02)) for p in prices]
    volumes = [np.random.uniform(50, 200) for _ in prices]

    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "value": [p * v for p, v in zip(prices, volumes)],
    })


class TestBacktestEngine:
    def setup_method(self):
        self.config = TradingConfig(
            grid_levels=5,
            grid_spacing_atr_mult=0.5,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            stop_loss_pct=0.05,
            max_drawdown_pct=0.10,
            min_order_krw=5_000,
        )
        self.engine = BacktestEngine(self.config)

    def test_basic_run(self):
        """백테스트가 정상 실행되어야 한다."""
        df = _make_candle_df(500)
        result = self.engine.run(df, "KRW-TEST", 500_000)

        assert result.market == "KRW-TEST"
        assert result.initial_balance == 500_000
        assert result.final_balance > 0
        assert len(result.equity_curve) > 0

    def test_empty_data(self):
        """빈 데이터 처리."""
        result = self.engine.run(pd.DataFrame(), "KRW-TEST", 500_000)
        assert result.final_balance == 500_000
        assert result.total_trades == 0

    def test_fees_deducted(self):
        """수수료가 차감되어야 한다."""
        df = _make_candle_df(500)
        result = self.engine.run(df, "KRW-TEST", 500_000)
        # 거래가 발생했다면 수수료도 있어야 함
        if result.total_trades > 0:
            assert result.total_fees > 0

    def test_drawdown_limit(self):
        """최대 드로다운이 한도를 크게 초과하면 안 된다."""
        df = _make_candle_df(500)
        result = self.engine.run(df, "KRW-TEST", 500_000)
        # 드로다운 한도(10%) + 약간의 여유 (청산 과정의 슬리피지)
        assert result.max_drawdown_pct < 0.20

    def test_win_rate_calculation(self):
        """승률 계산이 올바른지 확인."""
        df = _make_candle_df(500)
        result = self.engine.run(df, "KRW-TEST", 500_000)

        if result.total_trades > 0:
            expected_wr = result.winning_trades / result.total_trades
            assert abs(result.win_rate - expected_wr) < 0.01

    def test_low_volatility_data(self):
        """낮은 변동성 데이터에서도 안전하게 실행."""
        df = _make_candle_df(500, volatility=0.001)
        result = self.engine.run(df, "KRW-TEST", 500_000)
        assert result.final_balance > 0

    def test_high_volatility_data(self):
        """높은 변동성 데이터에서도 안전하게 실행."""
        df = _make_candle_df(500, volatility=0.05)
        result = self.engine.run(df, "KRW-TEST", 500_000)
        assert result.final_balance > 0
