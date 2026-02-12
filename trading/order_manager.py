"""주문 생성/추적/취소 관리. 그리드 레벨별 주문 일괄 처리."""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from core.api_client import UpbitClient
from data.database import TradeDB
from strategy.grid import Grid, GridLevel, GridSide, GridStatus
from trading.risk_manager import RiskManager
from trading.portfolio import Portfolio
from utils.notifier import Notifier
from utils.logger import get_trade_logger


class OrderManager:
    """주문 관리자. 그리드 기반 지정가 주문 생성/추적."""

    def __init__(
        self,
        client: UpbitClient,
        db: TradeDB,
        risk_manager: RiskManager,
        portfolio: Portfolio,
        notifier: Notifier | None = None,
        paper_mode: bool = False,
    ) -> None:
        self._client = client
        self._db = db
        self._risk = risk_manager
        self._portfolio = portfolio
        self._notifier = notifier or Notifier()
        self._paper_mode = paper_mode
        self._trade_log = get_trade_logger()

    async def place_grid_orders(
        self,
        grid: Grid,
        order_krw_per_level: float,
    ) -> Grid:
        """그리드 매수 주문 일괄 생성.

        각 매수 레벨에 지정가 주문을 등록하고, 상태를 ACTIVE로 변경.
        """
        for level in grid.buy_levels:
            if level.status != GridStatus.PENDING:
                continue

            # 주문 수량 계산
            volume = order_krw_per_level / level.price
            valid, reason = self._risk.validate_order(
                order_krw_per_level,
                await self._portfolio.get_available_krw(),
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
                    f"금액: {order_krw_per_level:,.0f}"
                )
                continue

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

                await self._db.save_trade({
                    "order_uuid": result["uuid"],
                    "market": grid.market,
                    "side": "bid",
                    "ord_type": "limit",
                    "price": level.price,
                    "volume": volume,
                    "total_krw": order_krw_per_level,
                    "grid_level": level.level,
                    "created_at": datetime.now().isoformat(),
                    "status": "wait",
                })

                self._trade_log.info(
                    f"매수 주문 | {grid.market} L{level.level} | "
                    f"가격: {level.price:,.0f} | 수량: {volume:.8f}"
                )

            except Exception as e:
                logger.error(f"매수 주문 실패 (레벨 {level.level}): {e}")

            # Rate limit 준수
            await asyncio.sleep(0.15)

        return grid

    async def place_sell_order(
        self, grid: Grid, buy_level: GridLevel
    ) -> GridLevel | None:
        """매수 체결 시 대응 매도 주문 생성.

        매수 레벨 N 체결 → 매도 레벨 N (= 매수가 + 1 spacing) 주문.
        """
        # 대응되는 매도 레벨 찾기
        sell_level = None
        for level in grid.sell_levels:
            if level.level == buy_level.level:
                sell_level = level
                break

        if sell_level is None:
            logger.error(f"대응 매도 레벨을 찾을 수 없음: {buy_level.level}")
            return None

        volume = buy_level.volume

        if self._paper_mode:
            sell_level.status = GridStatus.ACTIVE
            sell_level.volume = volume
            sell_level.order_uuid = f"paper-sell-{grid.market}-{sell_level.level}"
            self._trade_log.info(
                f"[PAPER] 매도 주문 | {grid.market} L{sell_level.level} | "
                f"가격: {sell_level.price:,.0f} | 수량: {volume:.8f}"
            )
            return sell_level

        try:
            result = await self._client.place_order(
                market=grid.market,
                side="ask",
                volume=f"{volume:.8f}",
                price=str(int(sell_level.price)),
                ord_type="limit",
            )
            sell_level.order_uuid = result["uuid"]
            sell_level.status = GridStatus.ACTIVE
            sell_level.volume = volume

            await self._db.save_trade({
                "order_uuid": result["uuid"],
                "market": grid.market,
                "side": "ask",
                "ord_type": "limit",
                "price": sell_level.price,
                "volume": volume,
                "total_krw": sell_level.price * volume,
                "grid_level": sell_level.level,
                "created_at": datetime.now().isoformat(),
                "status": "wait",
                "related_order_uuid": buy_level.order_uuid,
            })

            self._trade_log.info(
                f"매도 주문 | {grid.market} L{sell_level.level} | "
                f"가격: {sell_level.price:,.0f} | 수량: {volume:.8f}"
            )
            self._notifier.trade_executed("ask", grid.market, sell_level.price, volume)
            return sell_level

        except Exception as e:
            logger.error(f"매도 주문 실패: {e}")
            return None

    async def sync_orders(self, grid: Grid) -> Grid:
        """주문 상태 동기화. 체결된 매수 → 자동 매도 주문 생성."""
        for level in grid.levels:
            if level.status != GridStatus.ACTIVE or not level.order_uuid:
                continue

            if self._paper_mode:
                # Paper 모드: ticker 가격으로 체결 시뮬레이션
                continue

            try:
                order = await self._client.get_order(level.order_uuid)
                state = order.get("state", "")

                if state == "done":
                    # 체결 완료
                    level.status = GridStatus.FILLED
                    await self._db.update_trade_status(
                        level.order_uuid, "done", datetime.now().isoformat()
                    )

                    if level.side == GridSide.BUY:
                        # 매수 체결 → 포지션 기록 + 매도 주문 생성
                        await self._db.save_position({
                            "market": grid.market,
                            "entry_price": level.price,
                            "volume": level.volume,
                            "grid_level": level.level,
                            "stop_loss_price": grid.stop_loss_price,
                            "entry_time": datetime.now().isoformat(),
                        })
                        self._notifier.trade_executed(
                            "bid", grid.market, level.price, level.volume
                        )
                        await self.place_sell_order(grid, level)

                    elif level.side == GridSide.SELL:
                        # 매도 체결 → 포지션 청산 기록
                        self._trade_log.info(
                            f"매도 체결 | {grid.market} L{level.level} | "
                            f"가격: {level.price:,.0f}"
                        )

                elif state == "cancel":
                    level.status = GridStatus.CANCELLED
                    await self._db.update_trade_status(level.order_uuid, "cancel")

            except Exception as e:
                logger.error(f"주문 동기화 실패 ({level.order_uuid}): {e}")

            await asyncio.sleep(0.15)

        return grid

    async def check_paper_fills(self, grid: Grid, current_price: float) -> Grid:
        """Paper 모드: 현재가로 체결 시뮬레이션."""
        for level in grid.levels:
            if level.status != GridStatus.ACTIVE:
                continue

            filled = False
            if level.side == GridSide.BUY and current_price <= level.price:
                filled = True
            elif level.side == GridSide.SELL and current_price >= level.price:
                filled = True

            if filled:
                level.status = GridStatus.FILLED
                self._trade_log.info(
                    f"[PAPER] 체결 | {grid.market} L{level.level} "
                    f"{'매수' if level.side == GridSide.BUY else '매도'} | "
                    f"주문가: {level.price:,.0f} | 현재가: {current_price:,.0f}"
                )

                if level.side == GridSide.BUY:
                    await self.place_sell_order(grid, level)

        return grid

    async def cancel_all_orders(self, market: str) -> int:
        """특정 마켓 미체결 주문 전체 취소. 취소된 수 반환."""
        if self._paper_mode:
            logger.info(f"[PAPER] {market} 전체 주문 취소")
            return 0

        cancelled = 0
        try:
            open_orders = await self._client.get_open_orders(market)
            for order in open_orders:
                try:
                    await self._client.cancel_order(order["uuid"])
                    await self._db.update_trade_status(order["uuid"], "cancel")
                    cancelled += 1
                    await asyncio.sleep(0.15)
                except Exception as e:
                    logger.error(f"주문 취소 실패 ({order['uuid']}): {e}")

            logger.info(f"{market} 주문 {cancelled}건 취소")
        except Exception as e:
            logger.error(f"미체결 주문 조회 실패 ({market}): {e}")

        return cancelled

    async def emergency_sell(self, market: str) -> bool:
        """긴급 시장가 매도 (손절/드로다운 시)."""
        position = await self._portfolio.get_position(market)
        if not position or position.balance <= 0:
            return True

        if self._paper_mode:
            self._trade_log.info(
                f"[PAPER] 긴급 매도 | {market} | "
                f"수량: {position.balance:.8f} | 현재가: {position.current_price:,.0f}"
            )
            return True

        try:
            await self._client.place_order(
                market=market,
                side="ask",
                volume=f"{position.balance:.8f}",
                ord_type="market",
            )
            self._trade_log.info(
                f"긴급 시장가 매도 | {market} | 수량: {position.balance:.8f}"
            )
            self._notifier.stop_loss_triggered(
                market, position.unrealized_pnl_pct
            )
            return True
        except Exception as e:
            logger.error(f"긴급 매도 실패 ({market}): {e}")
            return False
