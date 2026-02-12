"""파라미터 최적화기. 그리드 서치로 최적 파라미터 탐색.

Usage:
    python -m backtest.optimizer --market KRW-BTC --days 180
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from config.settings import TradingConfig
from core.api_client import UpbitClient
from backtest.engine import BacktestEngine, BacktestResult


@dataclass
class ParamSet:
    """탐색할 파라미터 조합."""
    grid_spacing_atr_mult: float
    stop_loss_pct: float
    rsi_oversold: float
    grid_levels: int
    bb_std: float
    regime_adx_threshold: float


@dataclass
class OptResult:
    """최적화 결과."""
    params: ParamSet
    result: BacktestResult
    score: float


def compute_score(r: BacktestResult) -> float:
    """종합 스코어 계산.

    가중치:
    - 수익률 (30%): 양수 수익 보상
    - 승률 (25%): 70% 이상 목표
    - 손익비 (25%): 평균 이익 / 평균 손실
    - 샤프비율 (10%): 리스크 대비 수익
    - 낙폭 패널티 (10%): 큰 낙폭은 감점
    """
    if r.total_trades < 5:
        return -999.0  # 거래 부족 → 의미 없음

    # 수익률 점수
    return_score = r.total_return_pct * 100  # -5% → -5, +10% → +10

    # 승률 점수 (0~100)
    wr_score = r.win_rate * 100

    # 손익비 (Profit Factor)
    winning_pnl = sum(t.pnl for t in r.trades if t.pnl > 0)
    losing_pnl = abs(sum(t.pnl for t in r.trades if t.pnl < 0))
    profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 10.0

    # 샤프비율 점수
    sharpe_score = max(min(r.sharpe_ratio, 5.0), -5.0)

    # 낙폭 패널티
    dd_penalty = r.max_drawdown_pct * 100  # 5% → 5점 감점

    score = (
        return_score * 0.30
        + wr_score * 0.25
        + profit_factor * 10 * 0.25  # 1.0 → 2.5, 2.0 → 5.0
        + sharpe_score * 0.10
        - dd_penalty * 0.10
    )

    return round(score, 4)


def run_single(
    df: pd.DataFrame,
    market: str,
    balance: float,
    params: ParamSet,
) -> OptResult:
    """단일 파라미터 조합 백테스트."""
    config = TradingConfig()
    config.grid_spacing_atr_mult = params.grid_spacing_atr_mult
    config.stop_loss_pct = params.stop_loss_pct
    config.rsi_oversold = params.rsi_oversold
    config.grid_levels = params.grid_levels
    config.bb_std = params.bb_std
    config.regime_adx_threshold = params.regime_adx_threshold

    engine = BacktestEngine(config)
    result = engine.run(df, market, balance)
    score = compute_score(result)

    return OptResult(params=params, result=result, score=score)


def generate_param_grid() -> list[ParamSet]:
    """탐색할 파라미터 그리드 생성."""
    grid = {
        "grid_spacing_atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
        "stop_loss_pct": [0.03, 0.05, 0.07],
        "rsi_oversold": [25.0, 30.0, 35.0],
        "grid_levels": [3, 5],
        "bb_std": [1.5, 2.0, 2.5],
        "regime_adx_threshold": [20.0, 25.0, 30.0],
    }

    combos = list(product(
        grid["grid_spacing_atr_mult"],
        grid["stop_loss_pct"],
        grid["rsi_oversold"],
        grid["grid_levels"],
        grid["bb_std"],
        grid["regime_adx_threshold"],
    ))

    return [
        ParamSet(
            grid_spacing_atr_mult=c[0],
            stop_loss_pct=c[1],
            rsi_oversold=c[2],
            grid_levels=c[3],
            bb_std=c[4],
            regime_adx_threshold=c[5],
        )
        for c in combos
    ]


def print_results(results: list[OptResult], top_n: int = 15) -> None:
    """상위 결과 출력."""
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    valid = [r for r in sorted_results if r.result.total_trades >= 5]

    print("\n" + "=" * 110)
    print("  파라미터 최적화 결과 (상위 {}개)".format(min(top_n, len(valid))))
    print("=" * 110)
    print(
        f"{'순위':>4} | {'ATR배수':>7} | {'손절%':>5} | {'RSI':>5} | {'레벨':>4} | {'BB':>4} | {'ADX':>4} | "
        f"{'수익률':>8} | {'승률':>6} | {'손익비':>6} | {'거래':>4} | {'낙폭%':>6} | {'샤프':>6} | {'점수':>8}"
    )
    print("-" * 110)

    for rank, opt in enumerate(valid[:top_n], 1):
        p = opt.params
        r = opt.result
        winning_pnl = sum(t.pnl for t in r.trades if t.pnl > 0)
        losing_pnl = abs(sum(t.pnl for t in r.trades if t.pnl < 0))
        pf = winning_pnl / losing_pnl if losing_pnl > 0 else 99.9

        print(
            f"{rank:>4} | {p.grid_spacing_atr_mult:>7.1f} | {p.stop_loss_pct:>5.2f} | "
            f"{p.rsi_oversold:>5.0f} | {p.grid_levels:>4} | {p.bb_std:>4.1f} | {p.regime_adx_threshold:>4.0f} | "
            f"{r.total_return_pct:>+7.2%} | {r.win_rate:>5.1%} | {pf:>6.2f} | "
            f"{r.total_trades:>4} | {r.max_drawdown_pct:>5.2%} | {r.sharpe_ratio:>6.2f} | {opt.score:>8.2f}"
        )

    # 최적 파라미터 추천
    if valid:
        best = valid[0]
        p = best.params
        print("\n" + "=" * 110)
        print("  최적 파라미터 추천")
        print("=" * 110)
        print(f"  grid_spacing_atr_mult: {p.grid_spacing_atr_mult}")
        print(f"  stop_loss_pct:         {p.stop_loss_pct}")
        print(f"  rsi_oversold:          {p.rsi_oversold}")
        print(f"  grid_levels:           {p.grid_levels}")
        print(f"  bb_std:                {p.bb_std}")
        print(f"  regime_adx_threshold:  {p.regime_adx_threshold}")
        print(f"\n  수익률: {best.result.total_return_pct:+.2%}")
        print(f"  승률:   {best.result.win_rate:.1%}")
        r = best.result
        wp = sum(t.pnl for t in r.trades if t.pnl > 0)
        lp = abs(sum(t.pnl for t in r.trades if t.pnl < 0))
        print(f"  손익비: {wp / lp if lp > 0 else 99.9:.2f}")
        print(f"  거래수: {best.result.total_trades}")
        print(f"  낙폭:   {best.result.max_drawdown_pct:.2%}")
        print(f"  샤프:   {best.result.sharpe_ratio:.2f}")
        print("=" * 110)


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Parameter Optimizer")
    parser.add_argument("--market", default="KRW-BTC", help="마켓 코드")
    parser.add_argument("--days", type=int, default=180, help="백테스트 기간(일)")
    parser.add_argument("--balance", type=float, default=1_000_000, help="초기 잔고")
    args = parser.parse_args()

    from utils.logger import setup_logger
    setup_logger()

    # 1. 데이터 한 번만 수집
    config = TradingConfig()
    client = UpbitClient(config.access_key, config.secret_key)
    try:
        candles_needed = args.days * 24 * 60 // config.candle_unit
        df = await client.get_candles_extended(
            args.market, config.candle_unit, min(candles_needed, 10000)
        )
    finally:
        await client.close()

    if df.empty:
        logger.error("데이터 수집 실패")
        return

    logger.info(f"데이터 수집 완료: {len(df)}개 캔들 ({args.days}일)")

    # 2. 파라미터 그리드 생성
    param_grid = generate_param_grid()
    total = len(param_grid)
    logger.info(f"총 {total}개 파라미터 조합 탐색 시작")

    # 3. 백테스트 실행
    results: list[OptResult] = []
    start = time.time()

    for idx, params in enumerate(param_grid):
        opt_result = run_single(df, args.market, args.balance, params)
        results.append(opt_result)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            eta = elapsed / (idx + 1) * (total - idx - 1)
            logger.info(
                f"진행: {idx + 1}/{total} ({(idx + 1) / total:.0%}) | "
                f"경과: {elapsed:.0f}s | 남은: {eta:.0f}s"
            )

    elapsed = time.time() - start
    logger.info(f"최적화 완료: {total}개 조합, {elapsed:.1f}초 소요")

    # 4. 결과 출력
    print_results(results)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
