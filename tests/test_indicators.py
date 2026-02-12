"""기술적 지표 계산 테스트."""

import numpy as np
import pandas as pd
import pytest

from strategy.indicators import TechnicalIndicators


def _make_df(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """테스트용 캔들 DataFrame 생성."""
    n = len(prices)
    if volumes is None:
        volumes = [100.0] * n
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": volumes,
        "value": [p * v for p, v in zip(prices, volumes)],
    })


class TestRSI:
    def test_rsi_range(self):
        """RSI는 0~100 범위여야 한다."""
        prices = list(np.random.uniform(90000, 110000, 100))
        df = _make_df(prices)
        rsi = TechnicalIndicators.rsi(df, period=14)
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_rsi_oversold_on_decline(self):
        """지속적 하락 시 RSI < 30 이하여야 한다."""
        prices = [100000 - i * 500 for i in range(50)]
        df = _make_df(prices)
        rsi = TechnicalIndicators.rsi(df, period=14)
        assert rsi.iloc[-1] < 30

    def test_rsi_overbought_on_rise(self):
        """지속적 상승 시 RSI > 70 이상이어야 한다."""
        prices = [100000 + i * 2000 for i in range(50)]
        df = _make_df(prices)
        rsi = TechnicalIndicators.rsi(df, period=14)
        assert rsi.iloc[-1] > 70


class TestBollingerBands:
    def test_bands_order(self):
        """상단 > 중간 > 하단 밴드."""
        prices = list(np.random.uniform(90000, 110000, 50))
        df = _make_df(prices)
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df)

        valid = upper.dropna() > lower.dropna()
        assert valid.all()

    def test_price_within_bands_mostly(self):
        """대부분의 가격은 볼린저 밴드 내에 있어야 한다."""
        np.random.seed(42)
        prices = list(np.random.normal(100000, 1000, 100))
        df = _make_df(prices)
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df)

        within = ((df["close"] <= upper) & (df["close"] >= lower)).sum()
        total_valid = upper.notna().sum()
        ratio = within / total_valid if total_valid > 0 else 0
        assert ratio > 0.90  # 90% 이상 밴드 내


class TestATR:
    def test_atr_positive(self):
        """ATR은 항상 양수."""
        prices = list(np.random.uniform(90000, 110000, 50))
        df = _make_df(prices)
        atr = TechnicalIndicators.atr(df)
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_atr_reflects_volatility(self):
        """변동성이 큰 데이터의 ATR이 더 커야 한다."""
        low_vol = list(np.random.uniform(99000, 101000, 50))
        high_vol = list(np.random.uniform(90000, 110000, 50))

        atr_low = TechnicalIndicators.atr(_make_df(low_vol)).iloc[-1]
        atr_high = TechnicalIndicators.atr(_make_df(high_vol)).iloc[-1]
        assert atr_high > atr_low


class TestMACD:
    def test_macd_output_shape(self):
        """MACD는 3개의 시리즈를 반환."""
        prices = list(np.random.uniform(90000, 110000, 50))
        df = _make_df(prices)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(df)

        assert len(macd_line) == len(df)
        assert len(signal_line) == len(df)
        assert len(histogram) == len(df)

    def test_histogram_equals_diff(self):
        """히스토그램 = MACD - Signal."""
        prices = list(np.random.uniform(90000, 110000, 50))
        df = _make_df(prices)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(df)

        diff = macd_line - signal_line
        np.testing.assert_array_almost_equal(histogram.values, diff.values, decimal=6)


class TestCalculateAll:
    def test_all_columns_present(self):
        """calculate_all이 모든 지표 컬럼을 추가해야 한다."""
        prices = list(np.random.uniform(90000, 110000, 50))
        df = _make_df(prices)
        result = TechnicalIndicators.calculate_all(df)

        expected_cols = [
            "rsi", "bb_upper", "bb_middle", "bb_lower",
            "atr", "macd", "macd_signal", "macd_hist",
            "volume_ma", "bb_pct_b",
        ]
        for col in expected_cols:
            assert col in result.columns, f"{col} 컬럼 누락"
