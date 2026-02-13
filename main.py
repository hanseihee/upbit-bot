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

import pandas as pd
from loguru import logger

from config.settings import TradingConfig
from core.api_client import UpbitClient
from data.candle_fetcher import CandleFetcher
from data.database import TradeDB
from strategy.coin_selector import CoinSelector
from strategy.grid import AdaptiveGrid, Grid, GridStatus, GridSide
from strategy.indicators import TechnicalIndicators
from strategy.momentum import MomentumPosition, MomentumStatus
from strategy.signals import SignalEngine, SignalType, MarketRegime
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
        self._notifier = Notifier(
            bot_token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
            enabled=config.telegram_enabled,
        )
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
            adx_period=config.adx_period,
            regime_bb_width_low=config.regime_bb_width_low,
            regime_bb_width_high=config.regime_bb_width_high,
            regime_adx_threshold=config.regime_adx_threshold,
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

        # Paper 모드 잔고 추적
        self._paper_available_krw: float = config.paper_balance
        self._paper_total_value: float = config.paper_balance

        # 그리드 리사이클 쿨다운 추적 (market → 완료 시각)
        self._grid_completed_at: dict[str, datetime] = {}

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
            self._risk.set_initial_balance(self._config.paper_balance)
            logger.info(f"[PAPER] 초기 잔고: {self._config.paper_balance:,.0f} KRW")

        # 코인 초기 선별
        await self._select_coins()

        # 시작 알림
        balance = self._config.paper_balance if self._paper_mode else await self._portfolio.get_total_value()
        self._notifier.bot_started(mode_str, balance, self._selected_coins)

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

    def _get_available_krw(self) -> float:
        """Paper 모드 가용 잔고 반환."""
        return self._paper_available_krw

    async def _main_loop(self) -> None:
        """주기 메인 루프."""

        # 1. 잔고 확인 + 드로다운 체크 (Paper 모드 포함)
        if self._paper_mode:
            # Paper 모드: 그리드 포지션 가치 추산
            current_total = self._paper_available_krw
            # 활성 그리드의 체결된 매수 레벨 가치 추산은 단순화
            risk_check = self._risk.check_drawdown(current_total)
            if risk_check.should_close_all:
                logger.error(f"[PAPER] 드로다운 한도 초과! {risk_check.reason}")
                await self._close_all_positions()
                self._running = False
                return
        else:
            summary = await self._portfolio.get_summary()
            risk_check = self._risk.check_drawdown(summary.total_value)

            if risk_check.should_close_all:
                logger.error(f"드로다운 한도 초과! {risk_check.reason}")
                await self._close_all_positions()
                self._running = False
                return

            self._portfolio.log_summary(summary)

        # 2. 코인 재선별
        await self._maybe_reselect_coins()

        if not self._selected_coins:
            logger.warning("선별된 코인 없음, 대기 중...")
            return

        # 3. 데이터 수집
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

    def _is_grid_completed(self, grid: Grid) -> bool:
        """그리드가 완료되었는지 확인 (모든 활성 매도 체결 완료)."""
        filled_sells = [l for l in grid.sell_levels if l.status == GridStatus.FILLED and l.volume > 0]
        active_orders = [l for l in grid.levels if l.status == GridStatus.ACTIVE]
        has_volume_sells = any(l.volume > 0 for l in grid.sell_levels)
        return bool(filled_sells) and not active_orders and has_volume_sells

    def _update_adaptive_stop_loss(self, grid: Grid) -> None:
        """체결된 레벨 기준으로 손절가 동적 조정."""
        filled_buys = [l for l in grid.buy_levels if l.status == GridStatus.FILLED]
        if filled_buys:
            lowest_filled = min(l.price for l in filled_buys)
            new_stop = self._grid_engine._round_to_tick(
                lowest_filled * (1 - self._config.stop_loss_pct), lowest_filled
            )
            grid.stop_loss_price = new_stop

    def _is_recycle_cooldown_passed(self, market: str) -> bool:
        """그리드 리사이클 쿨다운 경과 여부."""
        if market not in self._grid_completed_at:
            return True
        elapsed = datetime.now() - self._grid_completed_at[market]
        # 쿨다운 = candle_unit(분) × cooldown_candles
        cooldown_min = self._config.candle_unit * self._config.grid_recycle_cooldown_candles
        return elapsed >= timedelta(minutes=cooldown_min)

    async def _process_market(self, market: str, df: pd.DataFrame) -> None:
        """개별 마켓 처리 로직."""

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

            # [FIX] 적응형 손절가: 체결된 레벨 기준으로 갱신
            self._update_adaptive_stop_loss(grid)

            # 손절 체크
            if current_price <= grid.stop_loss_price:
                logger.warning(f"손절가 도달: {market} {current_price:,.0f} <= {grid.stop_loss_price:,.0f}")
                await self._order_mgr.cancel_all_orders(market)
                if not self._paper_mode:
                    await self._order_mgr.emergency_sell(market)
                del self._active_grids[market]
                return

            # [FIX] 그리드 완료 감지
            if self._is_grid_completed(grid):
                logger.info(f"그리드 완료! 모든 매도 체결: {market}")
                self._grid_completed_at[market] = datetime.now()
                del self._active_grids[market]
                # Section C로 폴스루하여 리사이클 진입 가능
            else:
                # ATR 변화로 그리드 재조정
                if pd.notna(current_atr) and self._grid_engine.should_update_grid(
                    current_atr, grid.atr_at_creation
                ):
                    logger.info(f"ATR 변화 감지, 그리드 재조정: {market}")
                    await self._order_mgr.cancel_all_orders(market)
                    if not self._paper_mode:
                        await self._order_mgr.emergency_sell(market)
                    del self._active_grids[market]
                    return  # [FIX] return 추가 - 좀비 그리드 방지

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

        # C.0: 그리드 리사이클 (완료 직후 재진입)
        if (
            self._config.grid_recycle_enabled
            and market in self._grid_completed_at
            and self._is_recycle_cooldown_passed(market)
        ):
            recycle_signal = self._signal_engine.generate_recycle_signal(
                df, self._config.grid_recycle_rsi_threshold, market
            )
            if recycle_signal.type == SignalType.BUY and recycle_signal.is_actionable:
                logger.info(
                    f"그리드 리사이클 | {market} | 신뢰도: {recycle_signal.confidence:.2f} | "
                    f"이유: {', '.join(recycle_signal.reasons)}"
                )
                await self._enter_grid(market, df, recycle_signal, current_price, current_atr)
                del self._grid_completed_at[market]
                return

        # C.1: Mean Reversion 신호 (우선)
        signal_result = self._signal_engine.generate_signal(df, market)

        if signal_result.type == SignalType.BUY and signal_result.is_actionable:
            logger.info(
                f"매수 신호 | {market} | 신뢰도: {signal_result.confidence:.2f} | "
                f"체제: {signal_result.regime.value} | "
                f"이유: {', '.join(signal_result.reasons)}"
            )
            await self._enter_grid(market, df, signal_result, current_price, current_atr)
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
        regime = signal_result.regime if signal_result else MarketRegime.TRANSITIONAL
        bb_lower = latest["bb_lower"]
        adx_val = latest.get("adx", 0)
        # 이전 완성 캔들 기준 거래량 (현재 캔들은 미완성)
        prev_vol = prev["volume"]
        prev_vol_ma = prev["volume_ma"]
        vol_ratio = prev_vol / prev_vol_ma if pd.notna(prev_vol_ma) and prev_vol_ma > 0 else 0
        logger.info(
            f"HOLD | {market} | "
            f"가격: {current_price:,.0f} | "
            f"RSI: {latest['rsi']:.1f} | "
            f"ADX: {adx_val:.1f} | "
            f"체제: {regime.value} | "
            f"BB하단: {bb_lower:,.0f} | "
            f"거래량비: {vol_ratio:.1f}x"
        )

    async def _enter_grid(
        self,
        market: str,
        df: pd.DataFrame,
        signal_result: object,
        current_price: float,
        current_atr: float,
    ) -> None:
        """그리드 진입 공통 로직."""
        # [FIX] Paper 모드 잔고 추적
        if self._paper_mode:
            available = self._get_available_krw()
        else:
            available = await self._portfolio.get_available_krw()

        order_per_level = self._risk.calculate_position_size(
            available, current_price, signal_result.confidence, self._config.grid_levels  # type: ignore[attr-defined]
        )

        if order_per_level <= 0:
            logger.debug(f"주문금액 부족, 진입 불가: {market}")
            return

        if pd.isna(current_atr) or current_atr <= 0:
            logger.warning(f"ATR 계산 불가: {market}")
            return

        grid = self._grid_engine.calculate_grid(
            market, current_price, current_atr, self._config.grid_levels
        )

        self._notifier.grid_created(market, self._config.grid_levels, current_price)

        # [FIX] 역피라미드 사이징: 레벨별 차등 배분
        grid = await self._place_pyramid_grid_orders(grid, order_per_level)
        self._active_grids[market] = grid

        # Paper 잔고 차감
        if self._paper_mode:
            total_allocated = sum(l.volume * l.price for l in grid.buy_levels if l.status == GridStatus.ACTIVE)
            self._paper_available_krw -= total_allocated

        await self._db.save_grid_state(market, grid.to_json(), current_atr)

    async def _place_pyramid_grid_orders(self, grid: Grid, base_order_krw: float) -> Grid:
        """역피라미드 방식으로 그리드 매수 주문 배치.

        상위 레벨(현재가 근처)에 더 많이, 하위 레벨(멀리)에 적게 배분.
        가중치: [0.35, 0.25, 0.20, 0.12, 0.08] (5레벨 기준)
        """
        levels = len(grid.buy_levels)
        if levels <= 0:
            return grid

        # 역피라미드 가중치 생성 (합=1.0)
        raw_weights = [1.0 / (i + 1) for i in range(levels)]
        total_weight = sum(raw_weights)
        weights = [w / total_weight for w in raw_weights]

        # 총 투자금 = base_order_krw × levels (원래 균등 배분 시 총액)
        total_budget = base_order_krw * levels

        for i, level in enumerate(grid.buy_levels):
            if level.status != GridStatus.PENDING:
                continue

            level_krw = total_budget * weights[i]

            # 최소 주문금액 체크
            if level_krw < self._config.min_order_krw:
                continue

            volume = level_krw / level.price
            valid, reason = self._risk.validate_order(
                level_krw,
                self._get_available_krw() if self._paper_mode else await self._portfolio.get_available_krw(),
            )

            if not valid:
                logger.warning(f"주문 검증 실패 (레벨 {level.level}): {reason}")
                continue

            level.volume = volume

            if self._paper_mode:
                level.status = GridStatus.ACTIVE
                level.order_uuid = f"paper-buy-{grid.market}-{level.level}"
                self._trade_log.info(
                    f"[PAPER] 매수 주문 | {grid.market} L{level.level} | "
                    f"가격: {level.price:,.0f} | 수량: {volume:.8f} | "
                    f"금액: {level_krw:,.0f} | 가중치: {weights[i]:.1%}"
                )
            else:
                try:
                    result = await self._client.place_order(
                        market=grid.market,
                        side="bid",
                        volume=f"{volume:.8f}",
                        price=str(int(level.price)),
                        ord_type="limit",
                    )
                    level.order_uuid = result["uuid"]
                    level.status = GridStatus.ACTIVE

                    self._trade_log.info(
                        f"매수 주문 | {grid.market} L{level.level} | "
                        f"가격: {level.price:,.0f} | 수량: {volume:.8f} | "
                        f"가중치: {weights[i]:.1%}"
                    )
                except Exception as e:
                    logger.error(f"매수 주문 실패 (레벨 {level.level}): {e}")

                await asyncio.sleep(0.15)

        return grid

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
        signal_obj: object,
        current_price: float,
        current_atr: float,
    ) -> None:
        """모멘텀 포지션 진입. 시장가 매수."""
        # [FIX] Paper 모드 잔고 추적
        if self._paper_mode:
            available = self._get_available_krw()
        else:
            available = await self._portfolio.get_available_krw()

        order_krw = self._risk.calculate_momentum_size(
            available, signal_obj.confidence  # type: ignore[attr-defined]
        )
        if order_krw <= 0:
            logger.debug(f"모멘텀 주문금액 부족: {market}")
            return

        volume = order_krw / current_price

        logger.info(
            f"모멘텀 진입 | {market} | 가격: {current_price:,.0f} | "
            f"금액: {order_krw:,.0f} KRW | 신뢰도: {signal_obj.confidence:.2f} | "  # type: ignore[attr-defined]
            f"이유: {', '.join(signal_obj.reasons)}"  # type: ignore[attr-defined]
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
            self._paper_available_krw -= order_krw
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

        self._notifier.momentum_entry(market, current_price, order_krw)

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
            # Paper 잔고 복원 + 손익 반영
            self._paper_available_krw += pos.order_krw + pnl_krw
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

        self._notifier.momentum_exit(market, current_price, pnl_krw, pnl_pct, reason)

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
        """포지션 유지시간 초과 정리."""
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
        self._notifier.bot_stopped()
        await asyncio.sleep(1)  # 텔레그램 전송 대기
        await self._notifier.close()
        await self._db.close()
        await self._client.close()
        logger.info("봇 종료 완료")

    def stop(self) -> None:
        """외부에서 종료 요청."""
        self._running = False


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
