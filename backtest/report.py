"""백테스트 결과 리포트 생성."""

from __future__ import annotations

from loguru import logger


class BacktestReport:
    """백테스트 결과를 포맷팅하여 출력."""

    def __init__(self, result) -> None:
        self._r = result

    def print_summary(self) -> None:
        """콘솔에 요약 리포트 출력."""
        r = self._r
        sep = "=" * 55

        print(f"\n{sep}")
        print(f"  백테스트 결과 리포트")
        print(f"{sep}")
        print(f"  마켓:       {r.market}")
        print(f"  기간:       {r.start_date[:10]} ~ {r.end_date[:10]}")
        print(f"  초기 잔고:  {r.initial_balance:>15,.0f} KRW")
        print(f"  최종 잔고:  {r.final_balance:>15,.0f} KRW")
        print(f"{'-' * 55}")
        print(f"  총 수익:    {r.total_return:>+15,.0f} KRW ({r.total_return_pct:+.2%})")
        print(f"  승률:       {r.win_rate:>15.1%}")
        print(f"  총 거래:    {r.total_trades:>15d}건")
        print(f"  승/패:      {r.winning_trades:>7d} / {r.losing_trades:<7d}")
        print(f"  평균 손익:  {r.avg_trade_pnl:>+15,.0f} KRW")
        print(f"  총 수수료:  {r.total_fees:>15,.0f} KRW")
        print(f"{'-' * 55}")
        print(f"  최대 낙폭:  {r.max_drawdown:>15,.0f} KRW ({r.max_drawdown_pct:.2%})")
        print(f"  샤프비율:   {r.sharpe_ratio:>15.2f}")
        print(f"{sep}\n")

        # 승률 평가
        if r.win_rate >= 0.70:
            print("  [평가] 목표 승률(70%) 달성!")
        elif r.win_rate >= 0.60:
            print("  [평가] 승률 양호 (60-70%), 파라미터 미세 조정 권장")
        else:
            print("  [평가] 승률 미달 (<60%), 전략 재검토 필요")

        if r.max_drawdown_pct > 0.10:
            print(f"  [경고] 최대 낙폭이 10% 초과 ({r.max_drawdown_pct:.1%})")

        # 상위 수익/손실 거래
        if r.trades:
            sorted_trades = sorted(r.trades, key=lambda t: t.pnl, reverse=True)

            print(f"\n  Top 3 수익 거래:")
            for t in sorted_trades[:3]:
                if t.pnl > 0:
                    print(f"    {t.entry_price:,.0f} → {t.exit_price:,.0f} | PnL: {t.pnl:+,.0f} ({t.pnl_pct:+.2%})")

            print(f"\n  Top 3 손실 거래:")
            for t in sorted_trades[-3:]:
                if t.pnl < 0:
                    print(f"    {t.entry_price:,.0f} → {t.exit_price:,.0f} | PnL: {t.pnl:+,.0f} ({t.pnl_pct:+.2%})")

        print()

    def to_dict(self) -> dict:
        """리포트 데이터를 dict로 반환."""
        r = self._r
        return {
            "market": r.market,
            "period": f"{r.start_date[:10]} ~ {r.end_date[:10]}",
            "initial_balance": r.initial_balance,
            "final_balance": r.final_balance,
            "total_return": r.total_return,
            "total_return_pct": r.total_return_pct,
            "win_rate": r.win_rate,
            "total_trades": r.total_trades,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "total_fees": r.total_fees,
        }
