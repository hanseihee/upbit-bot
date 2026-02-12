"""Upbit 자동 트레이딩 봇 메인 루프.

Usage:
    python main.py --mode paper    # Paper Trading
    python main.py --mode live     # 실전 매매
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

from config.settings import TradingConfig
from core.api_client import UpbitClient
from data.candle_fetcher import CandleFetcher
from data.database import TradeDB
from strategy.coin_selector import CoinSelector
from strategy.grid import AdaptiveGrid, Grid, GridStatus, GridSide
from strategy.indicators import TechnicalIndicators
from strategy.momentum import MomentumPosition, MomentumStatus
from strategy.signals import SignalEngine, SignalType
from trading.order_manager import OrderManager
from trading.portfolio import Portfolio
from trading.risk_manager import RiskManager
from utils.logger import setup_logger, get_trade_logger
from utils.notifier import Notifier


class TradingBot:
    """메인 트레이딩 봇."""

    def __init__(self, config: TradingConfig, paper_mode: bool = True) -> None:
        self._config = config
        self._paper_mode = paper_mode
        self._running = False

        # 컴포넌트 초기화
        self._client = UpbitClient(
            config.access_key, config.secret_key, config.api_url
        )
        self._db = TradeDB(config.db_path)
        self._notifier = Notifier()
        self._fetcher = CandleFetcher(self._client)
        self._portfolio = Portfolio(self._client)
        self._risk = RiskManager(config, self._notifier)
        self._order_mgr = OrderManager(
            self._client, self._db, self._risk, self._portfolio,
            self._notifier, paper_mode,
        )
        self._coin_selector = CoinSelector(
            self._client,
            config.min_volume_krw,
            config.max_daily_change,
            config.max_spread_pct,
        )
        self._signal_engine = SignalEngine(
            rsi_oversold=config.rsi_oversold,
            rsi_overbought=config.rsi_overbought,
            bb_period=config.bb_period,
            bb_std=config.bb_std,
            rsi_period=config.rsi_period,
            atr_period=config.atr_period,
            volume_ma_period=config.volume_ma_period,
        )
        self._grid_engine = AdaptiveGrid(
            atr_multiplier=config.grid_spacing_atr_mult,
            stop_loss_pct=config.stop_loss_pct,
            fee_rate=config.upbit_fee_rate,
        )

        # 상태
        self._active_grids: dict[str, Grid] = {}
        self._momentum_positions: dict[str, MomentumPosition] = {}
        self._selected_coins: list[str] = []
        self._last_coin_select: datetime | None = None
        self._trade_log = get_trade_logger()

    async def start(self) -> None:
        """봇 시작."""
        mode_str = "PAPER" if self._paper_mode else "LIVE"
        logger.info(f"{'='*50}")
        logger.info(f"Upbit Trading Bot 시작 [{mode_str} 모드]")
        logger.info(f"{'='*50}")

        # 설정 검증
        if not self._paper_mode:
            errors = self._config.validate()
            if errors:
                for e in errors:
                    logger.error(e)
                return

        # DB 초기화
        await self._db.initialize()

        # 초기 잔고 설정
        if not self._paper_mode:
            total = await self._portfolio.get_total_value()
            self._risk.set_initial_balance(total)
        else:
            self._risk.set_initial_balance(500_000)  # Paper: 50만원
            logger.info("[PAPER] 초기 잔고: 500,000 KRW")

        # 코인 초기 선별
        await self._select_coins()

        # 메인 루프
        self._running = True
        while self._running:
            try:
                await self._main_loop()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"메인 루프 오류: {e}")
                await asyncio.sleep(5)

            await asyncio.sleep(self._config.check_interval_sec)

        await self._shutdown()

    async def _main_loop(self) -> None:
        """1분 주기 메인 루프."""

        # 1. 잔고 확인 + 드로다운 체크
        if not self._paper_mode:
            summary = await self._portfolio.get_summary()
            risk_check = self._risk.check_drawdown(summary.total_value)

            if risk_check.should_close_all:
                logger.error(f"드로다운 한도 초과! {risk_check.reason}")
                await self._close_all_positions()
                self._running = False
                return

            self._portfolio.log_summary(summary)

        # 2. 코인 재선별 (4시간마다)
        await self._maybe_reselect_coins()

        if not self._selected_coins:
            logger.warning("선별된 코인 없음, 대기 중...")
            return

        # 3. 데이터 수집 (병렬)
        logger.info(f"데이터 수집 중: {', '.join(self._selected_coins)}")
        candle_data = await self._fetcher.fetch_multiple(
            self._selected_coins, self._config.candle_unit, 200
        )

        # 4~7. 코인별 처리
        for market in self._selected_coins:
            df = candle_data.get(market)
            if df is None or df.empty:
                logger.warning(f"캔들 데이터 없음: {market}")
                continue

            try:
                await self._process_market(market, df)
            except Exception as e:
                logger.error(f"마켓 처리 오류 ({market}): {e}")

        # 8. 포지션 유지시간 체크
        await self._check_expired_positions()

        # 9. 활성 그리드 현황
        if self._active_grids:
            for m, g in self._active_grids.items():
                active_buys = sum(1 for l in g.buy_levels if l.status == GridStatus.ACTIVE)
                filled_buys = sum(1 for l in g.buy_levels if l.status == GridStatus.FILLED)
                logger.info(f"그리드 | {m} | 대기매수: {active_buys} | 체결매수: {filled_buys}")

        # 10. 모멘텀 포지션 현황
        if self._momentum_positions:
            for m, pos in self._momentum_positions.items():
                pnl_pct = (
                    (pos.highest_price - pos.entry_price) / pos.entry_price
                    if pos.entry_price > 0
                    else 0
                )
                logger.info(
                    f"모멘텀 | {m} | 진입: {pos.entry_price:,.0f} | "
                    f"고점: {pos.highest_price:,.0f} | "
                    f"트레일링: {pos.trailing_stop_price:,.0f} | "
                    f"PnL: {pnl_pct:+.2%}"
                )

        logger.info(f"다음 체크까지 {self._config.check_interval_sec}초 대기...")

    async def _process_market(self, market: str, df: pd.DataFrame) -> None:
        """개별 마켓 처리 로직."""
        import pandas as pd

        # 지표 계산
        analyzed = TechnicalIndicators.calculate_all(
            df,
            rsi_period=self._config.rsi_period,
            bb_period=self._config.bb_period,
            bb_std=self._config.bb_std,
            atr_period=self._config.atr_period,
            vol_period=self._config.volume_ma_period,
        )

        latest = analyzed.iloc[-1]
        prev = analyzed.iloc[-2]
        current_price = latest["close"]
        current_atr = latest["atr"]

        # ── A. 기존 그리드 관리 ──
        if market in self._active_grids:
            grid = self._active_grids[market]

            # 주문 동기화
            if self._paper_mode:
                grid = await self._order_mgr.check_paper_fills(grid, current_price)
            else:
                grid = await self._order_mgr.sync_orders(grid)

            # 손절 체크
            if current_price <= grid.stop_loss_price:
                logger.warning(f"손절가 도달: {market} {current_price:,.0f} <= {grid.stop_loss_price:,.0f}")
                await self._order_mgr.cancel_all_orders(market)
                if not self._paper_mode:
                    await self._order_mgr.emergency_sell(market)
                del self._active_grids[market]
                return

            # ATR 변화로 그리드 재조정
            if pd.notna(current_atr) and self._grid_engine.should_update_grid(
                current_atr, grid.atr_at_creation
            ):
                logger.info(f"ATR 변화 감지, 그리드 재조정: {market}")
                await self._order_mgr.cancel_all_orders(market)
                del self._active_grids[market]
                # 아래에서 새 그리드 생성

            # RSI 과매수 이익 실현
            rsi_val = latest["rsi"]
            if rsi_val > self._config.rsi_overbought:
                logger.info(f"RSI 과매수 ({rsi_val:.1f}), 이익 실현: {market}")
                await self._order_mgr.cancel_all_orders(market)
                if not self._paper_mode:
                    await self._order_mgr.emergency_sell(market)
                del self._active_grids[market]
                return

            self._active_grids[market] = grid
            return

        # ── B. 모멘텀 포지션 관리 ──
        if market in self._momentum_positions:
            await self._manage_momentum_position(market, latest, prev, current_price, current_atr)
            return

        # ── C. 새로운 진입 신호 확인 ──

        # C.1: Mean Reversion 신호 (우선)
        signal = self._signal_engine.generate_signal(df, market)

        if signal.type == SignalType.BUY and signal.is_actionable:
            logger.info(
                f"매수 신호 | {market} | 신뢰도: {signal.confidence:.2f} | "
                f"이유: {', '.join(signal.reasons)}"
            )

            # 포지션 사이징
            available = await self._portfolio.get_available_krw() if not self._paper_mode else 500_000
            order_per_level = self._risk.calculate_position_size(
                available, current_price, signal.confidence, self._config.grid_levels
            )

            if order_per_level <= 0:
                logger.debug(f"주문금액 부족, 진입 불가: {market}")
                return

            # 그리드 생성
            if pd.isna(current_atr) or current_atr <= 0:
                logger.warning(f"ATR 계산 불가: {market}")
                return

            grid = self._grid_engine.calculate_grid(
                market, current_price, current_atr, self._config.grid_levels
            )

            self._notifier.grid_created(market, self._config.grid_levels, current_price)

            # 그리드 매수 주문
            grid = await self._order_mgr.place_grid_orders(grid, order_per_level)
            self._active_grids[market] = grid

            # 그리드 상태 DB 저장
            await self._db.save_grid_state(market, grid.to_json(), current_atr)
            return

        # C.2: 모멘텀 신호 (Mean Reversion 미발동 시)
        if self._config.momentum_enabled:
            momentum_signal = self._signal_engine.generate_momentum_signal(df, market)
            if (
                momentum_signal.type == SignalType.MOMENTUM_BUY
                and momentum_signal.is_actionable
                and self._can_open_momentum()
            ):
                await self._enter_momentum_position(
                    market, momentum_signal, current_price, current_atr
                )
                return

        # C.3: 신호 없음
        bb_lower = latest["bb_lower"]
        vol = latest["volume"]
        vol_ma = latest["volume_ma"]
        vol_ratio = vol / vol_ma if pd.notna(vol_ma) and vol_ma > 0 else 0
        logger.info(
            f"HOLD | {market} | "
            f"가격: {current_price:,.0f} | "
            f"RSI: {latest['rsi']:.1f} | "
            f"BB하단: {bb_lower:,.0f} | "
            f"거래량비: {vol_ratio:.1f}x"
        )

    # ── 모멘텀 전략 메서드 ─────────────────────────────

    def _can_open_momentum(self) -> bool:
        """모멘텀 포지션 개수 제한 확인."""
        active = sum(
            1 for p in self._momentum_positions.values()
            if p.status == MomentumStatus.ACTIVE
        )
        return active < self._config.momentum_max_positions

    async def _enter_momentum_position(
        self,
        market: str,
        signal: object,
        current_price: float,
        current_atr: float,
    ) -> None:
        """모멘텀 포지션 진입. 시장가 매수."""
        available = (
            await self._portfolio.get_available_krw()
            if not self._paper_mode
            else 500_000
        )

        order_krw = self._risk.calculate_momentum_size(
            available, signal.confidence  # type: ignore[attr-defined]
        )
        if order_krw <= 0:
            logger.debug(f"모멘텀 주문금액 부족: {market}")
            return

        volume = order_krw / current_price

        logger.info(
            f"모멘텀 진입 | {market} | 가격: {current_price:,.0f} | "
            f"금액: {order_krw:,.0f} KRW | 신뢰도: {signal.confidence:.2f} | "  # type: ignore[attr-defined]
            f"이유: {', '.join(signal.reasons)}"  # type: ignore[attr-defined]
        )

        pos = MomentumPosition(
            market=market,
            entry_price=current_price,
            volume=volume,
            order_krw=order_krw,
            entry_time=datetime.now().isoformat(),
            highest_price=current_price,
            trailing_stop_pct=self._config.momentum_trailing_stop_pct,
            atr_at_entry=current_atr if pd.notna(current_atr) else 0.0,
            atr_stop_multiplier=self._config.momentum_atr_stop_mult,
            hard_stop_pct=self._config.momentum_hard_stop_pct,
            rsi_exit=self._config.momentum_rsi_exit,
            max_hold_minutes=self._config.momentum_max_hold_minutes,
        )

        if self._paper_mode:
            pos.status = MomentumStatus.ACTIVE
            pos.order_uuid = f"paper-momentum-{market}"
            self._trade_log.info(
                f"[PAPER] 모멘텀 매수 | {market} | "
                f"가격: {current_price:,.0f} | 수량: {volume:.8f} | "
                f"금액: {order_krw:,.0f}"
            )
        else:
            try:
                result = await self._client.place_order(
                    market=market,
                    side="bid",
                    price=str(int(order_krw)),
                    ord_type="price",  # Upbit 시장가 매수
                )
                pos.order_uuid = result["uuid"]
                pos.status = MomentumStatus.ACTIVE
            except Exception as e:
                logger.error(f"모멘텀 매수 실패 ({market}): {e}")
                return

        pos.update_trailing_stop(current_price, pos.atr_at_entry)
        self._momentum_positions[market] = pos

        self._notifier.notify(
            "모멘텀 진입",
            f"{market} | 가격: {current_price:,.0f} | 트레일링: {pos.trailing_stop_price:,.0f}",
        )

        await self._db.save_momentum_state(market, pos.to_json())

    async def _manage_momentum_position(
        self,
        market: str,
        latest: pd.Series,
        prev: pd.Series,
        current_price: float,
        current_atr: float,
    ) -> None:
        """모멘텀 포지션 관리. 트레일링 스톱 + 이탈 조건."""
        pos = self._momentum_positions[market]

        rsi = latest["rsi"]
        macd_hist = latest["macd_hist"]
        macd_hist_prev = prev["macd_hist"]

        atr = current_atr if pd.notna(current_atr) else pos.atr_at_entry

        should_exit, reason = pos.should_exit(
            current_price, atr, rsi, macd_hist, macd_hist_prev
        )

        if should_exit:
            logger.info(f"모멘텀 이탈 | {market} | 이유: {reason}")
            await self._exit_momentum_position(market, current_price, reason)
            return

        new_stop = pos.update_trailing_stop(current_price, atr)

        pnl_pct = (
            (current_price - pos.entry_price) / pos.entry_price
            if pos.entry_price > 0
            else 0
        )
        logger.info(
            f"모멘텀 | {market} | "
            f"현재: {current_price:,.0f} | "
            f"고점: {pos.highest_price:,.0f} | "
            f"트레일링: {new_stop:,.0f} | "
            f"PnL: {pnl_pct:+.2%}"
        )

    async def _exit_momentum_position(
        self, market: str, current_price: float, reason: str
    ) -> None:
        """모멘텀 포지션 청산."""
        pos = self._momentum_positions[market]
        pnl_krw, pnl_pct = pos.calculate_pnl(current_price)

        if self._paper_mode:
            self._trade_log.info(
                f"[PAPER] 모멘텀 매도 | {market} | "
                f"가격: {current_price:,.0f} | "
                f"PnL: {pnl_krw:+,.0f} KRW ({pnl_pct:+.2%}) | "
                f"이유: {reason}"
            )
        else:
            try:
                await self._client.place_order(
                    market=market,
                    side="ask",
                    volume=f"{pos.volume:.8f}",
                    ord_type="market",
                )
            except Exception as e:
                logger.error(f"모멘텀 매도 실패 ({market}): {e}")
                return

        result_icon = "수익" if pnl_pct > 0 else "손실"
        self._notifier.notify(
            f"모멘텀 청산 ({result_icon})",
            f"{market} | PnL: {pnl_krw:+,.0f} KRW ({pnl_pct:+.2%}) | {reason}",
        )

        await self._db.close_momentum_state(market)
        del self._momentum_positions[market]

    async def _select_coins(self) -> None:
        """코인 선별. fixed_coins 설정 시 고정 코인 사용."""
        if self._config.fixed_coins:
            self._selected_coins = self._config.fixed_coins
            logger.info(f"고정 코인 사용: {self._selected_coins}")
        else:
            self._selected_coins = await self._coin_selector.select_coins(
                self._config.max_coins
            )
            logger.info(f"선별 코인: {self._selected_coins}")
        self._last_coin_select = datetime.now()

    async def _maybe_reselect_coins(self) -> None:
        """재선별 시간 확인 후 실행."""
        if self._last_coin_select is None:
            await self._select_coins()
            return

        elapsed = datetime.now() - self._last_coin_select
        if elapsed >= timedelta(hours=self._config.coin_reselect_hours):
            # 기존 그리드가 없는 경우에만 재선별
            if not self._active_grids:
                await self._select_coins()

    async def _check_expired_positions(self) -> None:
        """48시간 초과 포지션 정리."""
        expired = await self._db.get_expired_positions(
            self._config.max_position_age_hours
        )
        for pos in expired:
            market = pos["market"]
            logger.warning(f"포지션 유지시간 초과: {market}")

            if market in self._active_grids:
                await self._order_mgr.cancel_all_orders(market)
                del self._active_grids[market]

            if not self._paper_mode:
                await self._order_mgr.emergency_sell(market)

            await self._db.close_position(
                pos["id"], pos.get("entry_price", 0), 0
            )

    async def _close_all_positions(self) -> None:
        """전체 포지션 청산."""
        logger.error("전체 포지션 청산 시작!")

        # 그리드 포지션 청산
        for market in list(self._active_grids.keys()):
            await self._order_mgr.cancel_all_orders(market)
            if not self._paper_mode:
                await self._order_mgr.emergency_sell(market)
        self._active_grids.clear()

        # 모멘텀 포지션 청산
        for market in list(self._momentum_positions.keys()):
            if not self._paper_mode:
                await self._order_mgr.emergency_sell(market)
            await self._db.close_momentum_state(market)
        self._momentum_positions.clear()

        logger.error("전체 포지션 청산 완료")

    async def _shutdown(self) -> None:
        """봇 종료."""
        logger.info("봇 종료 중...")
        self._running = False
        await self._db.close()
        await self._client.close()
        logger.info("봇 종료 완료")

    def stop(self) -> None:
        """외부에서 종료 요청."""
        self._running = False


# pandas import (process_market에서 사용)
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upbit Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="실행 모드 (기본: paper)",
    )
    return parser.parse_args()


async def async_main() -> None:
    config = TradingConfig()
    args = parse_args()

    setup_logger(config.log_dir)

    paper_mode = args.mode == "paper"
    bot = TradingBot(config, paper_mode=paper_mode)

    # 시그널 핸들러
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.stop)
        except NotImplementedError:
            # Windows에서는 signal handler가 제한적
            pass

    await bot.start()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("키보드 인터럽트로 종료")


if __name__ == "__main__":
    main()
