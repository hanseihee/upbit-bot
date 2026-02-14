"""여러 파라미터 조합을 한번에 백테스트 비교."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import TradingConfig
from backtest.engine import BacktestEngine


CONFIGS = {
    # F 기준 (이전 라운드 최선)
    "F0: ATR4 ST1.5 기준": dict(
        grid_levels=3, grid_spacing_atr_mult=4.0, stop_loss_pct=0.010,
        rsi_oversold=42.0, min_signal_confidence=0.35, max_per_coin_ratio=0.40,
        grid_sell_target_mult=1.5,
    ),
    # F 변형: sell target 높이기
    "F1: ATR4 ST2.0": dict(
        grid_levels=3, grid_spacing_atr_mult=4.0, stop_loss_pct=0.010,
        rsi_oversold=42.0, min_signal_confidence=0.35, max_per_coin_ratio=0.40,
        grid_sell_target_mult=2.0,
    ),
    # F 변형: 더 넓은 간격
    "F2: ATR5 ST1.5": dict(
        grid_levels=3, grid_spacing_atr_mult=5.0, stop_loss_pct=0.010,
        rsi_oversold=42.0, min_signal_confidence=0.35, max_per_coin_ratio=0.40,
        grid_sell_target_mult=1.5,
    ),
    # F 변형: 더 넓은 간격 + 높은 ST
    "F3: ATR5 ST2.0": dict(
        grid_levels=3, grid_spacing_atr_mult=5.0, stop_loss_pct=0.010,
        rsi_oversold=42.0, min_signal_confidence=0.35, max_per_coin_ratio=0.40,
        grid_sell_target_mult=2.0,
    ),
    # F 변형: 신호 좀 더 엄격
    "F4: ATR4 ST1.5 엄격": dict(
        grid_levels=3, grid_spacing_atr_mult=4.0, stop_loss_pct=0.010,
        rsi_oversold=40.0, min_signal_confidence=0.38, max_per_coin_ratio=0.40,
        grid_sell_target_mult=1.5,
    ),
    # F 변형: 2레벨 집중
    "F5: 2lv ATR4 ST1.5": dict(
        grid_levels=2, grid_spacing_atr_mult=4.0, stop_loss_pct=0.010,
        rsi_oversold=42.0, min_signal_confidence=0.35, max_per_coin_ratio=0.50,
        grid_sell_target_mult=1.5,
    ),
    # F 변형: 손절 넓힘
    "F6: ATR4 ST1.5 SL1.5%": dict(
        grid_levels=3, grid_spacing_atr_mult=4.0, stop_loss_pct=0.015,
        rsi_oversold=42.0, min_signal_confidence=0.35, max_per_coin_ratio=0.40,
        grid_sell_target_mult=1.5,
    ),
    # F 변형: ATR6 초광폭
    "F7: ATR6 ST2.0": dict(
        grid_levels=3, grid_spacing_atr_mult=6.0, stop_loss_pct=0.012,
        rsi_oversold=42.0, min_signal_confidence=0.35, max_per_coin_ratio=0.40,
        grid_sell_target_mult=2.0,
    ),
}


async def main():
    base = TradingConfig()
    engine0 = BacktestEngine(base)
    df = await engine0.fetch_data("KRW-BTC", 7)
    if df.empty:
        print("데이터 수집 실패")
        return

    print(f"\n{'='*100}")
    print(f"  F계열 최적화 백테스트 (7일, KRW-BTC, 잔고 1,000,000)")
    print(f"{'='*100}")
    print(f"{'설정':<25} {'거래수':>6} {'승률':>7} {'수익률':>8} {'수수료':>9} {'MDD':>7} {'승/패':>10} {'평균PnL':>9}")
    print(f"{'-'*100}")

    for name, overrides in CONFIGS.items():
        cfg = TradingConfig(**{**{
            k: getattr(base, k) for k in vars(base)
            if not k.startswith('_')
        }, **overrides})
        engine = BacktestEngine(cfg)
        r = engine.run(df, "KRW-BTC", 1_000_000)
        print(
            f"{name:<25} {r.total_trades:>6d} {r.win_rate:>6.1%} "
            f"{r.total_return_pct:>+7.2%} {r.total_fees:>9,.0f} "
            f"{r.max_drawdown_pct:>6.2%} {r.winning_trades:>4d}/{r.losing_trades:<4d}  "
            f"{r.avg_trade_pnl:>+8,.0f}"
        )

    print(f"{'='*100}\n")


if __name__ == "__main__":
    asyncio.run(main())
