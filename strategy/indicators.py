"""기술적 지표 계산 모듈. pandas 기반 순수 계산 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """기술적 지표 계산 유틸리티."""

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI (Relative Strength Index) 계산.

        Args:
            df: 'close' 컬럼이 포함된 DataFrame
            period: RSI 기간 (기본 14)

        Returns:
            RSI 값 Series (0~100)
        """
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        # avg_loss가 0이면 RSI=100, avg_gain이 0이면 RSI=0
        rsi = pd.Series(np.where(
            avg_loss == 0,
            np.where(avg_gain == 0, 50.0, 100.0),
            100 - (100 / (1 + avg_gain / avg_loss)),
        ), index=df.index)
        return rsi.fillna(50)  # 데이터 부족 시 중립값

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame, period: int = 20, std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산.

        Returns:
            (상단밴드, 중간밴드, 하단밴드) 튜플
        """
        middle = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()
        upper = middle + (rolling_std * std)
        lower = middle - (rolling_std * std)
        return upper, middle, lower

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산."""
        high = df["high"]
        low = df["low"]
        close_prev = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    @staticmethod
    def macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD 계산.

        Returns:
            (MACD line, Signal line, Histogram) 튜플
        """
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """거래량 이동평균."""
        return df["volume"].rolling(window=period).mean()

    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        vol_period: int = 20,
    ) -> pd.DataFrame:
        """모든 지표를 한 번에 계산하여 DataFrame에 추가."""
        result = df.copy()

        result["rsi"] = TechnicalIndicators.rsi(df, rsi_period)

        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df, bb_period, bb_std)
        result["bb_upper"] = bb_upper
        result["bb_middle"] = bb_middle
        result["bb_lower"] = bb_lower

        result["atr"] = TechnicalIndicators.atr(df, atr_period)

        macd_line, signal_line, histogram = TechnicalIndicators.macd(df)
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_hist"] = histogram

        result["volume_ma"] = TechnicalIndicators.volume_ma(df, vol_period)

        # 볼린저 밴드 %b (0~1, 하단=0, 상단=1)
        band_width = bb_upper - bb_lower
        result["bb_pct_b"] = (df["close"] - bb_lower) / band_width.replace(0, np.nan)

        return result
