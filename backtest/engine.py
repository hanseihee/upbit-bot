"""백테스트 엔진. 과거 캔들 데이터로 그리드 전략 검증.

Usage:
    python -m backtest.engine --market KRW-BTC --days 90
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from loguru import logger

from config.settings import TradingConfig
from core.api_client import UpbitClient
from strategy.grid import AdaptiveGrid, Grid, GridSide, GridStatus
from strategy.indicators import TechnicalIndicators
from strategy.signals import SignalEngine, SignalType
from backtest.report import BacktestReport


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    pnl_pct: float
    fee: float


@dataclass
class BacktestResult:
    market: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_return: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_pnl: float = 0.0
    total_fees: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """과거 데이터 기반 전략 백테스트."""

    def __init__(self, config: TradingConfig | None = None) -> None:
        self._config = config or TradingConfig()
        self._signal_engine = SignalEngine(
            rsi_oversold=self._config.rsi_oversold,
            rsi_overbought=self._config.rsi_overbought,
            bb_period=self._config.bb_period,
            bb_std=self._config.bb_std,
            rsi_period=self._config.rsi_period,
            atr_period=self._config.atr_period,
            volume_ma_period=self._config.volume_ma_period,
            adx_period=self._config.adx_period,
            regime_bb_width_low=self._config.regime_bb_width_low,
            regime_bb_width_high=self._config.regime_bb_width_high,
            regime_adx_threshold=self._config.regime_adx_threshold,
        )
        self._grid_engine = AdaptiveGrid(
            atr_multiplier=self._config.grid_spacing_atr_mult,
            stop_loss_pct=self._config.stop_loss_pct,
            fee_rate=self._config.upbit_fee_rate,
        )

    async def fetch_data(self, market: str, days: int) -> pd.DataFrame:
        """과거 캔들 데이터 수집."""
        client = UpbitClient(
            self._config.access_key, self._config.secret_key
        )
        try:
            candles_needed = days * 24 * 60 // self._config.candle_unit
            df = await client.get_candles_extended(
                market, self._config.candle_unit, min(candles_needed, 10000)
            )
            logger.info(f"데이터 수집 완료: {market} {len(df)}개 캔들")
            return df
        finally:
            await client.close()

    def run(
        self,
        df: pd.DataFrame,
        market: str,
        initial_balance: float = 500_000,
    ) -> BacktestResult:
        """백테스트 실행.

        시뮬레이션:
        1. 각 캔들에서 신호 확인
        2. BUY 신호 → 그리드 생성 + 가상 주문
        3. 가격이 그리드 레벨 도달 시 체결 시뮬레이션
        4. 손절/이익실현 처리
        5. 수수료 반영 (0.05%)
        """
        if df.empty:
            logger.warning("데이터 없음")
            return BacktestResult(
                market=market,
                start_date="",
                end_date="",
                initial_balance=initial_balance,
                final_balance=initial_balance,
            )

        # 지표 계산
        analyzed = TechnicalIndicators.calculate_all(
            df,
            rsi_period=self._config.rsi_period,
            bb_period=self._config.bb_period,
            bb_std=self._config.bb_std,
            atr_period=self._config.atr_period,
            vol_period=self._config.volume_ma_period,
            adx_period=self._config.adx_period,
        )

        balance = initial_balance
        peak_balance = initial_balance
        max_drawdown = 0.0
        active_grid: Grid | None = None
        trades: list[Trade] = []
        equity_curve: list[float] = [initial_balance]
        total_fees = 0.0
        position_value = 0.0

        # 최소 데이터 요구 (지표 계산에 필요한 기간)
        start_idx = max(self._config.bb_period, 26) + 5

        for i in range(start_idx, len(analyzed)):
            row = analyzed.iloc[i]
            price = row["close"]
            high = row["high"]
            low = row["low"]
            atr = row["atr"]
            dt = str(row["datetime"])

            # 현재 자산 평가
            current_total = balance + position_value

            # 드로다운 체크
            if current_total > peak_balance:
                peak_balance = current_total
            dd = (peak_balance - current_total) / peak_balance if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

            # 드로다운 한도 초과 → 전체 청산
            if dd >= self._config.max_drawdown_pct and active_grid:
                for level in active_grid.buy_levels:
                    if level.status == GridStatus.FILLED:
                        fee = level.volume * price * self._config.upbit_fee_rate
                        total_fees += fee
                        pnl = (price - level.price) * level.volume - fee
                        trades.append(Trade(
                            entry_time=dt, exit_time=dt,
                            side="sell", entry_price=level.price,
                            exit_price=price, volume=level.volume,
                            pnl=pnl, pnl_pct=pnl / (level.price * level.volume) if level.volume > 0 else 0,
                            fee=fee,
                        ))
                        balance += level.volume * price - fee
                position_value = 0.0
                active_grid = None

            # 활성 그리드 처리
            if active_grid:
                # 매수 체결 확인 (저가가 매수 레벨 이하)
                just_filled_levels: set[int] = set()  # 동일봉 매수+매도 방지
                for level in active_grid.buy_levels:
                    if level.status == GridStatus.ACTIVE and low <= level.price:
                        level.status = GridStatus.FILLED
                        fee = level.volume * level.price * self._config.upbit_fee_rate
                        total_fees += fee
                        balance -= level.volume * level.price + fee
                        just_filled_levels.add(level.level)

                        # 대응 매도 레벨 활성화
                        for sl in active_grid.sell_levels:
                            if sl.level == level.level and sl.status == GridStatus.PENDING:
                                sl.status = GridStatus.ACTIVE
                                sl.volume = level.volume

                # 매도 체결 확인 (고가가 매도 레벨 이상)
                # 동일봉에 매수된 레벨은 매도 처리하지 않음 (look-ahead bias 방지)
                for level in active_grid.sell_levels:
                    if (level.status == GridStatus.ACTIVE
                            and level.volume > 0
                            and high >= level.price
                            and level.level not in just_filled_levels):
                        level.status = GridStatus.FILLED
                        fee = level.volume * level.price * self._config.upbit_fee_rate
                        total_fees += fee
                        sell_value = level.volume * level.price - fee
                        balance += sell_value

                        # 대응 매수 레벨 가격 찾기 + CANCELLED 처리 (이중 계산 방지)
                        entry_price = 0
                        for bl in active_grid.buy_levels:
                            if bl.level == level.level:
                                entry_price = bl.price
                                bl.status = GridStatus.CANCELLED  # 매도 완료 → 포지션 없음
                                break

                        pnl = (level.price - entry_price) * level.volume - fee * 2
                        trades.append(Trade(
                            entry_time=dt, exit_time=dt,
                            side="sell", entry_price=entry_price,
                            exit_price=level.price, volume=level.volume,
                            pnl=pnl,
                            pnl_pct=pnl / (entry_price * level.volume) if level.volume > 0 else 0,
                            fee=fee * 2,
                        ))

                # 손절 체크
                if low <= active_grid.stop_loss_price:
                    for level in active_grid.buy_levels:
                        if level.status == GridStatus.FILLED:
                            fee = level.volume * active_grid.stop_loss_price * self._config.upbit_fee_rate
                            total_fees += fee
                            pnl = (active_grid.stop_loss_price - level.price) * level.volume - fee
                            trades.append(Trade(
                                entry_time=dt, exit_time=dt,
                                side="stop_loss", entry_price=level.price,
                                exit_price=active_grid.stop_loss_price,
                                volume=level.volume, pnl=pnl,
                                pnl_pct=pnl / (level.price * level.volume) if level.volume > 0 else 0,
                                fee=fee,
                            ))
                            balance += level.volume * active_grid.stop_loss_price - fee
                    position_value = 0.0
                    active_grid = None

                # ATR 변화 체크 → 그리드 재조정
                elif pd.notna(atr) and active_grid and self._grid_engine.should_update_grid(
                    atr, active_grid.atr_at_creation
                ):
                    # 기존 포지션 정리
                    for level in active_grid.buy_levels:
                        if level.status == GridStatus.FILLED:
                            fee = level.volume * price * self._config.upbit_fee_rate
                            total_fees += fee
                            pnl = (price - level.price) * level.volume - fee
                            balance += level.volume * price - fee
                            trades.append(Trade(
                                entry_time=dt, exit_time=dt,
                                side="rebalance", entry_price=level.price,
                                exit_price=price, volume=level.volume,
                                pnl=pnl,
                                pnl_pct=pnl / (level.price * level.volume) if level.volume > 0 else 0,
                                fee=fee,
                            ))
                    position_value = 0.0
                    active_grid = None

                # 모든 레벨 체결 완료 → 그리드 종료
                elif active_grid:
                    all_sells_done = all(
                        l.status == GridStatus.FILLED
                        for l in active_grid.sell_levels
                        if l.volume > 0
                    )
                    no_active = not any(
                        l.status == GridStatus.ACTIVE
                        for l in active_grid.levels
                    )
                    if all_sells_done and no_active and any(l.volume > 0 for l in active_grid.sell_levels):
                        active_grid = None
                        position_value = 0.0

            # 신규 진입 확인 (그리드 없을 때만)
            if active_grid is None:
                window = analyzed.iloc[max(0, i - 200):i + 1]
                signal = self._signal_engine._evaluate_signal(window, market)

                if signal.type == SignalType.BUY and signal.is_actionable and pd.notna(atr) and atr > 0:
                    order_per_level = (
                        balance * signal.confidence
                        * self._config.max_per_coin_ratio
                        / self._config.grid_levels
                    )

                    if order_per_level >= self._config.min_order_krw:
                        grid = self._grid_engine.calculate_grid(
                            market, price, atr, self._config.grid_levels
                        )
                        # 매수 주문 설정
                        for level in grid.buy_levels:
                            level.volume = order_per_level / level.price
                            level.status = GridStatus.ACTIVE
                        active_grid = grid

            # 포지션 가치 업데이트
            if active_grid:
                position_value = sum(
                    l.volume * price
                    for l in active_grid.buy_levels
                    if l.status == GridStatus.FILLED
                )

            equity_curve.append(balance + position_value)

        # 결과 집계
        final_balance = balance + position_value
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        # 샤프비율 계산
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)) if returns.std() > 0 else 0
        else:
            sharpe = 0.0

        result = BacktestResult(
            market=market,
            start_date=str(analyzed.iloc[start_idx]["datetime"]),
            end_date=str(analyzed.iloc[-1]["datetime"]),
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_return=final_balance - initial_balance,
            total_return_pct=(final_balance - initial_balance) / initial_balance,
            win_rate=len(winning) / len(trades) if trades else 0,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            max_drawdown=max_drawdown * initial_balance,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=float(sharpe),
            avg_trade_pnl=sum(t.pnl for t in trades) / len(trades) if trades else 0,
            total_fees=total_fees,
            trades=trades,
            equity_curve=equity_curve,
        )

        return result


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Engine")
    parser.add_argument("--market", default="KRW-BTC", help="마켓 코드")
    parser.add_argument("--days", type=int, default=90, help="백테스트 기간(일)")
    parser.add_argument("--balance", type=float, default=500_000, help="초기 잔고")
    args = parser.parse_args()

    from utils.logger import setup_logger
    setup_logger()

    config = TradingConfig()
    engine = BacktestEngine(config)

    logger.info(f"백테스트 시작: {args.market} / {args.days}일 / 잔고 {args.balance:,.0f}")

    df = await engine.fetch_data(args.market, args.days)
    if df.empty:
        logger.error("데이터 수집 실패")
        return

    result = engine.run(df, args.market, args.balance)

    report = BacktestReport(result)
    report.print_summary()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
