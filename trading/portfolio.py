"""포트폴리오 상태 관리. 잔고, 포지션, 손익 통합 조회."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from core.api_client import UpbitClient


@dataclass
class Position:
    market: str
    currency: str
    balance: float        # 보유 수량
    avg_buy_price: float  # 평균 매수가
    current_price: float = 0.0

    @property
    def total_value(self) -> float:
        return self.balance * self.current_price

    @property
    def invested(self) -> float:
        return self.balance * self.avg_buy_price

    @property
    def unrealized_pnl(self) -> float:
        return self.total_value - self.invested

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.invested <= 0:
            return 0.0
        return self.unrealized_pnl / self.invested


@dataclass
class PortfolioSummary:
    total_krw: float = 0.0
    available_krw: float = 0.0
    positions_value: float = 0.0
    total_value: float = 0.0
    unrealized_pnl: float = 0.0
    positions: list[Position] = field(default_factory=list)


class Portfolio:
    """포트폴리오 관리자. Upbit API로 실시간 잔고/포지션 조회."""

    def __init__(self, client: UpbitClient) -> None:
        self._client = client
        self._last_summary: PortfolioSummary | None = None

    async def get_summary(self) -> PortfolioSummary:
        """포트폴리오 전체 요약 조회."""
        balances = await self._client.get_balances()
        summary = PortfolioSummary()

        coin_markets: list[str] = []
        coin_balances: dict[str, dict] = {}

        for b in balances:
            currency = b["currency"]
            balance = float(b["balance"]) + float(b.get("locked", 0))
            avg_price = float(b.get("avg_buy_price", 0))

            if currency == "KRW":
                summary.total_krw = balance
                summary.available_krw = float(b["balance"])
                continue

            if balance <= 0:
                continue

            market = f"KRW-{currency}"
            coin_markets.append(market)
            coin_balances[market] = {
                "currency": currency,
                "balance": balance,
                "avg_buy_price": avg_price,
            }

        # 현재가 조회
        if coin_markets:
            tickers = await self._client.get_ticker(coin_markets)
            ticker_map = {t["market"]: t for t in tickers}

            for market, info in coin_balances.items():
                current_price = 0.0
                if market in ticker_map:
                    current_price = ticker_map[market].get("trade_price", 0)

                pos = Position(
                    market=market,
                    currency=info["currency"],
                    balance=info["balance"],
                    avg_buy_price=info["avg_buy_price"],
                    current_price=current_price,
                )
                summary.positions.append(pos)
                summary.positions_value += pos.total_value
                summary.unrealized_pnl += pos.unrealized_pnl

        summary.total_value = summary.total_krw + summary.positions_value
        self._last_summary = summary
        return summary

    async def get_total_value(self) -> float:
        """총 평가금액."""
        summary = await self.get_summary()
        return summary.total_value

    async def get_available_krw(self) -> float:
        """주문 가능 KRW."""
        balances = await self._client.get_balances()
        for b in balances:
            if b["currency"] == "KRW":
                return float(b["balance"])
        return 0.0

    async def get_position(self, market: str) -> Position | None:
        """특정 마켓 포지션 조회."""
        summary = await self.get_summary()
        for pos in summary.positions:
            if pos.market == market:
                return pos
        return None

    async def has_position(self, market: str) -> bool:
        """특정 마켓 포지션 보유 여부."""
        pos = await self.get_position(market)
        return pos is not None and pos.balance > 0

    def log_summary(self, summary: PortfolioSummary) -> None:
        """포트폴리오 요약 로깅."""
        logger.info(
            f"포트폴리오 | 총평가: {summary.total_value:,.0f} KRW | "
            f"가용: {summary.available_krw:,.0f} KRW | "
            f"미실현 PnL: {summary.unrealized_pnl:+,.0f} KRW"
        )
        for pos in summary.positions:
            logger.info(
                f"  {pos.market}: {pos.balance:.8f} | "
                f"평단: {pos.avg_buy_price:,.0f} | "
                f"현재: {pos.current_price:,.0f} | "
                f"PnL: {pos.unrealized_pnl_pct:+.2%}"
            )
