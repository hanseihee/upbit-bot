"""Upbit REST API 비동기 클라이언트. Rate limit 자동 throttling 포함."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pandas as pd
from loguru import logger

from core.auth import UpbitAuth


class RateLimiter:
    """토큰 버킷 기반 rate limiter."""

    def __init__(self, max_calls: int, period: float = 1.0) -> None:
        self._max = max_calls
        self._period = period
        self._tokens = max_calls
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._max, self._tokens + elapsed * (self._max / self._period))
                self._last_refill = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return

                wait = (1 - self._tokens) * (self._period / self._max)
                self._tokens = 0

            # lock 밖에서 sleep → 다른 코루틴 블로킹 방지
            logger.debug(f"Rate limit 대기: {wait:.2f}s")
            await asyncio.sleep(wait)


class UpbitClient:
    """Upbit REST API 클라이언트."""

    def __init__(self, access_key: str, secret_key: str, api_url: str = "https://api.upbit.com/v1") -> None:
        self._auth = UpbitAuth(access_key, secret_key)
        self._base_url = api_url
        self._client = httpx.AsyncClient(timeout=15.0)
        # Upbit 실제 제한은 초당 10회이지만, 캔들 연속 조회 시 더 엄격하게 적용됨
        self._market_limiter = RateLimiter(5, 1.0)
        self._order_limiter = RateLimiter(8, 1.0)

    async def close(self) -> None:
        await self._client.aclose()

    # ── 시세 조회 (인증 불필요) ─────────────────────────

    async def _get_public(self, path: str, params: dict | None = None, max_retries: int = 5) -> Any:
        """공개 API 호출. 429 발생 시 지수 백오프로 재시도."""
        url = f"{self._base_url}{path}"

        for attempt in range(max_retries):
            await self._market_limiter.acquire()
            resp = await self._client.get(url, params=params)

            if resp.status_code == 429:
                # Remaining 헤더에서 대기시간 추출, 없으면 지수 백오프
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = min(2 ** attempt * 0.5, 30.0)
                logger.warning(f"429 Rate Limited (시도 {attempt + 1}/{max_retries}), {wait:.1f}초 대기")
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        # 최종 재시도까지 실패하면 마지막 응답 기준으로 예외 발생
        resp.raise_for_status()
        return resp.json()

    async def get_markets(self) -> list[dict]:
        """KRW 마켓 목록 조회."""
        markets = await self._get_public("/market/all", {"isDetails": "true"})
        return [m for m in markets if m["market"].startswith("KRW-")]

    async def get_candles(
        self, market: str, unit: int = 15, count: int = 200
    ) -> pd.DataFrame:
        """분봉 캔들 데이터 조회. DataFrame 반환."""
        path = f"/candles/minutes/{unit}"
        params = {"market": market, "count": min(count, 200)}
        data = await self._get_public(path, params)

        df = pd.DataFrame(data)
        if df.empty:
            return df

        df = df.rename(columns={
            "candle_date_time_kst": "datetime",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
            "candle_acc_trade_price": "value",
        })
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df[["datetime", "open", "high", "low", "close", "volume", "value"]]

    async def get_candles_extended(
        self, market: str, unit: int = 15, count: int = 500
    ) -> pd.DataFrame:
        """200개 초과 캔들 데이터를 페이징으로 수집."""
        all_data: list[pd.DataFrame] = []
        remaining = count
        to = None

        page = 0
        while remaining > 0:
            batch = min(remaining, 200)
            path = f"/candles/minutes/{unit}"
            params: dict[str, Any] = {"market": market, "count": batch}
            if to:
                params["to"] = to

            data = await self._get_public(path, params)
            if not data:
                break

            df = pd.DataFrame(data)
            all_data.append(df)
            remaining -= len(data)
            to = data[-1]["candle_date_time_utc"]
            page += 1

            if page % 10 == 0:
                logger.info(f"캔들 수집 진행: {count - remaining}/{count}")

            if len(data) < batch:
                break

            # 페이징 간 딜레이 — Upbit 429 방지
            await asyncio.sleep(0.2)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.rename(columns={
            "candle_date_time_kst": "datetime",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
            "candle_acc_trade_price": "value",
        })
        combined["datetime"] = pd.to_datetime(combined["datetime"])
        combined = combined.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
        return combined[["datetime", "open", "high", "low", "close", "volume", "value"]]

    async def get_ticker(self, markets: list[str]) -> list[dict]:
        """현재가 정보 조회."""
        return await self._get_public("/ticker", {"markets": ",".join(markets)})

    async def get_orderbook(self, markets: list[str]) -> list[dict]:
        """호가 정보 조회."""
        return await self._get_public("/orderbook", {"markets": ",".join(markets)})

    async def get_daily_candles(self, market: str, count: int = 7) -> pd.DataFrame:
        """일봉 조회 (코인 선별용)."""
        data = await self._get_public("/candles/days", {"market": market, "count": count})
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df = df.rename(columns={
            "candle_date_time_kst": "datetime",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
            "candle_acc_trade_price": "value",
        })
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)

    # ── 거래 (인증 필요) ───────────────────────────────

    async def _request_private(
        self, method: str, path: str, query: dict | None = None, body: dict | None = None,
        max_retries: int = 3,
    ) -> Any:
        """인증 API 호출. 429 발생 시 지수 백오프로 재시도."""
        url = f"{self._base_url}{path}"

        for attempt in range(max_retries):
            await self._order_limiter.acquire()
            headers = self._auth.get_auth_header(query or body)

            if method == "GET":
                resp = await self._client.get(url, params=query, headers=headers)
            elif method == "POST":
                resp = await self._client.post(url, json=body, headers=headers)
            elif method == "DELETE":
                resp = await self._client.delete(url, params=query, headers=headers)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")

            if resp.status_code == 429:
                wait = min(2 ** attempt * 0.5, 10.0)
                logger.warning(f"429 Rate Limited (private, 시도 {attempt + 1}/{max_retries}), {wait:.1f}초 대기")
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        resp.raise_for_status()
        return resp.json()

    async def get_balances(self) -> list[dict]:
        """계좌 잔고 조회."""
        return await self._request_private("GET", "/accounts")

    async def place_order(
        self,
        market: str,
        side: str,
        volume: str | None = None,
        price: str | None = None,
        ord_type: str = "limit",
    ) -> dict:
        """주문 생성.

        Args:
            market: 마켓 코드 (예: KRW-BTC)
            side: bid(매수) / ask(매도)
            volume: 주문량 (시장가 매수 시 None)
            price: 주문 가격 (시장가 매도 시 None, 시장가 매수 시 총액)
            ord_type: limit(지정가) / price(시장가 매수) / market(시장가 매도)
        """
        body: dict[str, str] = {
            "market": market,
            "side": side,
            "ord_type": ord_type,
        }
        if volume is not None:
            body["volume"] = str(volume)
        if price is not None:
            body["price"] = str(price)

        logger.info(f"주문 생성: {body}")
        return await self._request_private("POST", "/orders", body=body)

    async def cancel_order(self, uuid: str) -> dict:
        """주문 취소."""
        query = {"uuid": uuid}
        return await self._request_private("DELETE", "/order", query=query)

    async def get_order(self, uuid: str) -> dict:
        """주문 상세 조회."""
        query = {"uuid": uuid}
        return await self._request_private("GET", "/order", query=query)

    async def get_open_orders(self, market: str) -> list[dict]:
        """미체결 주문 목록 조회."""
        query = {"market": market, "state": "wait"}
        return await self._request_private("GET", "/orders", query=query)
