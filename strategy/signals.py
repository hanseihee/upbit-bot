"""매매 신호 생성 엔진. Mean Reversion 필터 + 신뢰도 계산."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
from loguru import logger

from strategy.indicators import TechnicalIndicators


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    MOMENTUM_BUY = "momentum_buy"


@dataclass
class Signal:
    type: SignalType
    confidence: float  # 0.0 ~ 1.0
    reasons: list[str]
    market: str = ""
    price: float = 0.0
    strategy: str = "mean_reversion"

    @property
    def is_actionable(self) -> bool:
        return self.type != SignalType.HOLD and self.confidence >= 0.6


class SignalEngine:
    """매매 신호 생성기. RSI + BB + Volume + MACD 기반."""

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        atr_period: int = 14,
        volume_ma_period: int = 20,
    ) -> None:
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._rsi_period = rsi_period
        self._atr_period = atr_period
        self._vol_period = volume_ma_period

    def generate_signal(self, df: pd.DataFrame, market: str = "") -> Signal:
        """캔들 데이터로부터 매매 신호 생성.

        진입 조건 (모두 충족 시 BUY):
        - RSI(14) < 30 (과매도)
        - 가격 <= BB 하단밴드
        - 현재 거래량 > 20일 평균 거래량
        - MACD 히스토그램 상승 전환 (보강)

        청산 조건 (하나라도 충족 시 SELL):
        - RSI(14) > 70 (과매수)
        - 가격 >= BB 상단밴드
        """
        if len(df) < max(self._bb_period, 26) + 5:
            return Signal(
                type=SignalType.HOLD,
                confidence=0.0,
                reasons=["데이터 부족"],
                market=market,
            )

        # 지표 계산
        analyzed = TechnicalIndicators.calculate_all(
            df,
            rsi_period=self._rsi_period,
            bb_period=self._bb_period,
            bb_std=self._bb_std,
            atr_period=self._atr_period,
            vol_period=self._vol_period,
        )

        latest = analyzed.iloc[-1]
        prev = analyzed.iloc[-2]
        price = latest["close"]

        # ── 매수 신호 평가 ──
        buy_reasons: list[str] = []
        buy_score = 0.0

        # RSI 과매도 (필수)
        rsi_val = latest["rsi"]
        if rsi_val < self._rsi_oversold:
            buy_reasons.append(f"RSI 과매도: {rsi_val:.1f}")
            buy_score += 0.30
        elif rsi_val < self._rsi_oversold + 5:
            buy_reasons.append(f"RSI 과매도 근접: {rsi_val:.1f}")
            buy_score += 0.10

        # BB 하단 이탈 (필수)
        bb_lower = latest["bb_lower"]
        if price <= bb_lower:
            buy_reasons.append(f"BB 하단 이탈: {price:,.0f} <= {bb_lower:,.0f}")
            buy_score += 0.30
        elif price <= bb_lower * 1.005:
            buy_reasons.append(f"BB 하단 근접: {price:,.0f}")
            buy_score += 0.10

        # 거래량 확인
        vol = latest["volume"]
        vol_ma = latest["volume_ma"]
        if pd.notna(vol_ma) and vol_ma > 0 and vol > vol_ma:
            buy_reasons.append(f"거래량 증가: {vol / vol_ma:.1f}x")
            buy_score += 0.20

        # MACD 히스토그램 상승 전환 (보강)
        macd_hist = latest["macd_hist"]
        macd_hist_prev = prev["macd_hist"]
        if pd.notna(macd_hist) and pd.notna(macd_hist_prev):
            if macd_hist > macd_hist_prev:
                buy_reasons.append("MACD 히스토그램 상승 전환")
                buy_score += 0.20

        # ── 매도 신호 평가 ──
        sell_reasons: list[str] = []
        sell_score = 0.0

        if rsi_val > self._rsi_overbought:
            sell_reasons.append(f"RSI 과매수: {rsi_val:.1f}")
            sell_score += 0.40

        bb_upper = latest["bb_upper"]
        if price >= bb_upper:
            sell_reasons.append(f"BB 상단 이탈: {price:,.0f} >= {bb_upper:,.0f}")
            sell_score += 0.40

        if pd.notna(macd_hist) and pd.notna(macd_hist_prev):
            if macd_hist < macd_hist_prev and macd_hist_prev > 0:
                sell_reasons.append("MACD 히스토그램 하락 전환")
                sell_score += 0.20

        # ── 최종 판단 ──
        if buy_score > sell_score and buy_score >= 0.5:
            return Signal(
                type=SignalType.BUY,
                confidence=min(buy_score, 1.0),
                reasons=buy_reasons,
                market=market,
                price=price,
            )
        elif sell_score > buy_score and sell_score >= 0.3:
            return Signal(
                type=SignalType.SELL,
                confidence=min(sell_score, 1.0),
                reasons=sell_reasons,
                market=market,
                price=price,
            )
        else:
            return Signal(
                type=SignalType.HOLD,
                confidence=0.0,
                reasons=["신호 강도 부족"],
                market=market,
                price=price,
            )

    def generate_momentum_signal(self, df: pd.DataFrame, market: str = "") -> Signal:
        """모멘텀/브레이크아웃 매수 신호 생성.

        진입 조건 (가중 스코어링):
        1. RSI > 55 & 상승 중        [0.20]
        2. 가격 > BB 상단             [0.25]
        3. MACD 히스토그램 양수 & 확대 [0.25]
        4. 거래량 > 1.5x 평균         [0.20]
        5. ATR 10% 이상 확대          [0.10]
        """
        if len(df) < max(self._bb_period, 26) + 5:
            return Signal(
                type=SignalType.HOLD,
                confidence=0.0,
                reasons=["데이터 부족"],
                market=market,
                strategy="momentum",
            )

        analyzed = TechnicalIndicators.calculate_all(
            df,
            rsi_period=self._rsi_period,
            bb_period=self._bb_period,
            bb_std=self._bb_std,
            atr_period=self._atr_period,
            vol_period=self._vol_period,
        )

        latest = analyzed.iloc[-1]
        prev = analyzed.iloc[-2]
        price = latest["close"]

        reasons: list[str] = []
        score = 0.0

        # 1. RSI 모멘텀
        rsi_val = latest["rsi"]
        rsi_prev = prev["rsi"]
        if pd.notna(rsi_val) and pd.notna(rsi_prev):
            if rsi_val > 55 and rsi_val > rsi_prev:
                reasons.append(f"RSI 상승 모멘텀: {rsi_val:.1f}")
                score += 0.20
            elif rsi_val > 50 and rsi_val > rsi_prev:
                reasons.append(f"RSI 약한 상승: {rsi_val:.1f}")
                score += 0.10

        # 2. BB 상단 돌파
        bb_upper = latest["bb_upper"]
        if pd.notna(bb_upper) and bb_upper > 0:
            if price > bb_upper:
                reasons.append(f"BB 상단 돌파: {price:,.0f} > {bb_upper:,.0f}")
                score += 0.25
            elif price > bb_upper * 0.998:
                reasons.append(f"BB 상단 근접: {price:,.0f}")
                score += 0.10

        # 3. MACD 히스토그램 양수 & 확대
        macd_hist = latest["macd_hist"]
        macd_hist_prev = prev["macd_hist"]
        if pd.notna(macd_hist) and pd.notna(macd_hist_prev):
            if macd_hist > 0 and macd_hist > macd_hist_prev:
                reasons.append(f"MACD 히스토그램 확대: {macd_hist:.2f}")
                score += 0.25
            elif macd_hist > 0:
                reasons.append(f"MACD 히스토그램 양수: {macd_hist:.2f}")
                score += 0.10

        # 4. 거래량 급증
        vol = latest["volume"]
        vol_ma = latest["volume_ma"]
        if pd.notna(vol_ma) and vol_ma > 0:
            vol_ratio = vol / vol_ma
            if vol_ratio > 1.5:
                reasons.append(f"거래량 급증: {vol_ratio:.1f}x")
                score += 0.20
            elif vol_ratio > 1.2:
                reasons.append(f"거래량 증가: {vol_ratio:.1f}x")
                score += 0.10

        # 5. ATR 확대
        atr_val = latest["atr"]
        atr_prev = prev["atr"]
        if pd.notna(atr_val) and pd.notna(atr_prev) and atr_prev > 0:
            atr_change = (atr_val - atr_prev) / atr_prev
            if atr_change > 0.10:
                reasons.append(f"ATR 확대: {atr_change:.1%}")
                score += 0.10

        # 과매수 페널티
        if pd.notna(rsi_val) and rsi_val > self._rsi_overbought:
            score *= 0.5
            reasons.append(f"RSI 과매수 주의: {rsi_val:.1f}")

        if score >= 0.60:
            return Signal(
                type=SignalType.MOMENTUM_BUY,
                confidence=min(score, 1.0),
                reasons=reasons,
                market=market,
                price=price,
                strategy="momentum",
            )

        return Signal(
            type=SignalType.HOLD,
            confidence=0.0,
            reasons=["모멘텀 신호 부족"],
            market=market,
            price=price,
            strategy="momentum",
        )

    def calculate_confidence(self, df: pd.DataFrame) -> float:
        """현재 시장 상태의 전반적 신뢰도 평가."""
        signal = self.generate_signal(df)
        return signal.confidence
