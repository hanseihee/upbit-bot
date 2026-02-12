"""캔들 데이터 수집 + 메모리 캐싱."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from core.api_client import UpbitClient


@dataclass
class _CacheEntry:
    data: pd.DataFrame
    timestamp: float


class CandleFetcher:
    """캔들 데이터를 캐싱하여 중복 API 호출 방지."""

    def __init__(self, client: UpbitClient, cache_ttl: float = 60.0) -> None:
        self._client = client
        self._cache_ttl = cache_ttl
        self._cache: dict[str, _CacheEntry] = {}

    def _cache_key(self, market: str, unit: int, count: int) -> str:
        return f"{market}:{unit}:{count}"

    async def fetch_candles(
        self, market: str, unit: int = 15, count: int = 200
    ) -> pd.DataFrame:
        """캔들 데이터 조회 (캐싱 적용)."""
        key = self._cache_key(market, unit, count)
        now = time.time()

        if key in self._cache:
            entry = self._cache[key]
            if now - entry.timestamp < self._cache_ttl:
                logger.debug(f"캐시 히트: {key}")
                return entry.data.copy()

        logger.debug(f"캔들 데이터 조회: {market} {unit}분봉 {count}개")

        if count > 200:
            df = await self._client.get_candles_extended(market, unit, count)
        else:
            df = await self._client.get_candles(market, unit, count)

        self._cache[key] = _CacheEntry(data=df, timestamp=now)
        return df.copy()

    async def fetch_multiple(
        self, markets: list[str], unit: int = 15, count: int = 200
    ) -> dict[str, pd.DataFrame]:
        """여러 코인 캔들 데이터 병렬 조회."""
        tasks = [self.fetch_candles(m, unit, count) for m in markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data: dict[str, pd.DataFrame] = {}
        for market, result in zip(markets, results):
            if isinstance(result, Exception):
                logger.error(f"캔들 조회 실패 ({market}): {result}")
                data[market] = pd.DataFrame()
            else:
                data[market] = result
        return data

    def clear_cache(self) -> None:
        """캐시 전체 초기화."""
        self._cache.clear()

    def evict_expired(self) -> int:
        """만료된 캐시 항목 제거. 제거된 수 반환."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v.timestamp >= self._cache_ttl]
        for k in expired:
            del self._cache[k]
        return len(expired)
