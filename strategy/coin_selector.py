"""자동 코인 선별 모듈. 거래량/변동성/스프레드 기준 스코어링."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from core.api_client import UpbitClient


@dataclass
class CoinScore:
    market: str
    korean_name: str
    volume_24h: float       # 24h 거래대금 (KRW)
    change_rate: float      # 일일 변동률
    spread_pct: float       # 스프레드 비율
    score: float = 0.0


class CoinSelector:
    """KRW 마켓에서 전략에 적합한 코인을 자동 선별."""

    def __init__(
        self,
        client: UpbitClient,
        min_volume_krw: float = 1_000_000_000,
        max_daily_change: float = 0.10,
        max_spread_pct: float = 0.003,
    ) -> None:
        self._client = client
        self._min_volume = min_volume_krw
        self._max_change = max_daily_change
        self._max_spread = max_spread_pct

    async def select_coins(self, max_coins: int = 2) -> list[str]:
        """최적 코인 선별 후 마켓 코드 리스트 반환.

        알고리즘:
        1. KRW 마켓 전체 ticker 조회
        2. 필터링: 거래대금, 변동률, 스프레드
        3. 스코어링: 거래량(0.4) + 변동성 적정범위(0.4) + 스프레드(0.2)
        4. 상위 N개 반환
        """
        markets = await self._client.get_markets()
        krw_codes = [m["market"] for m in markets]
        name_map = {m["market"]: m.get("korean_name", "") for m in markets}

        if not krw_codes:
            logger.warning("KRW 마켓이 없습니다")
            return []

        tickers = await self._client.get_ticker(krw_codes)
        orderbooks = await self._client.get_orderbook(krw_codes)

        # 호가 데이터를 마켓 기준으로 매핑
        ob_map: dict[str, dict] = {}
        for ob in orderbooks:
            ob_map[ob["market"]] = ob

        candidates: list[CoinScore] = []

        for ticker in tickers:
            market = ticker["market"]
            volume_24h = ticker.get("acc_trade_price_24h", 0)
            change_rate = abs(ticker.get("signed_change_rate", 0))
            trade_price = ticker.get("trade_price", 0)

            # 기본 필터
            if volume_24h < self._min_volume:
                continue
            if change_rate > self._max_change:
                continue
            if trade_price <= 0:
                continue

            # 스프레드 계산
            ob = ob_map.get(market)
            if not ob or not ob.get("orderbook_units"):
                continue

            best_ask = ob["orderbook_units"][0].get("ask_price", 0)
            best_bid = ob["orderbook_units"][0].get("bid_price", 0)
            if best_bid <= 0:
                continue

            spread_pct = (best_ask - best_bid) / best_bid
            if spread_pct > self._max_spread:
                continue

            candidates.append(CoinScore(
                market=market,
                korean_name=name_map.get(market, ""),
                volume_24h=volume_24h,
                change_rate=change_rate,
                spread_pct=spread_pct,
            ))

        if not candidates:
            logger.warning("필터 기준을 충족하는 코인이 없습니다")
            return []

        # 스코어링
        volumes = np.array([c.volume_24h for c in candidates])
        changes = np.array([c.change_rate for c in candidates])
        spreads = np.array([c.spread_pct for c in candidates])

        # 거래량: 높을수록 좋음 (정규화)
        vol_max = volumes.max() if volumes.max() > 0 else 1
        vol_scores = volumes / vol_max

        # 변동성: 3~7% 구간이 그리드에 최적 (너무 낮으면 수익 X, 높으면 위험)
        change_scores = np.where(
            changes < 0.03,
            changes / 0.03,  # 3% 미만: 선형 증가
            np.where(
                changes <= 0.07,
                1.0,  # 3~7%: 최적
                1.0 - (changes - 0.07) / 0.03,  # 7% 초과: 감소
            ),
        )
        change_scores = np.clip(change_scores, 0, 1)

        # 스프레드: 낮을수록 좋음
        spread_max = spreads.max() if spreads.max() > 0 else 1
        spread_scores = 1 - (spreads / spread_max)

        # 최종 스코어 = 거래량(0.4) + 변동성(0.4) + 스프레드(0.2)
        for i, c in enumerate(candidates):
            c.score = float(
                vol_scores[i] * 0.4
                + change_scores[i] * 0.4
                + spread_scores[i] * 0.2
            )

        # 정렬 및 상위 N개 선택
        candidates.sort(key=lambda x: x.score, reverse=True)
        selected = candidates[:max_coins]

        for c in selected:
            logger.info(
                f"선별: {c.market} ({c.korean_name}) | "
                f"거래대금: {c.volume_24h / 1e8:.0f}억 | "
                f"변동률: {c.change_rate:.2%} | "
                f"스프레드: {c.spread_pct:.4%} | "
                f"스코어: {c.score:.3f}"
            )

        return [c.market for c in selected]
