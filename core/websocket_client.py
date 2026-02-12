"""Upbit WebSocket 실시간 데이터 수신 클라이언트. 자동 재연결 포함."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Callable, Coroutine

import websockets
from loguru import logger

Callback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class UpbitWebSocket:
    """Upbit WebSocket 클라이언트."""

    def __init__(self, ws_url: str = "wss://api.upbit.com/websocket/v1") -> None:
        self._ws_url = ws_url
        self._ws: Any = None
        self._running = False
        self._callbacks: dict[str, list[Callback]] = {
            "ticker": [],
            "trade": [],
            "orderbook": [],
        }
        self._markets: list[str] = []
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    def on_ticker(self, callback: Callback) -> None:
        self._callbacks["ticker"].append(callback)

    def on_trade(self, callback: Callback) -> None:
        self._callbacks["trade"].append(callback)

    def on_orderbook(self, callback: Callback) -> None:
        self._callbacks["orderbook"].append(callback)

    async def subscribe(
        self, markets: list[str], types: list[str] | None = None
    ) -> None:
        """WebSocket 구독 시작."""
        self._markets = markets
        self._running = True
        subscribe_types = types or ["ticker", "trade", "orderbook"]

        while self._running:
            try:
                async with websockets.connect(self._ws_url, ping_interval=30) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    logger.info(f"WebSocket 연결 성공: {markets}")

                    # 구독 메시지 전송
                    subscribe_msg = [{"ticket": str(uuid.uuid4())}]
                    for stype in subscribe_types:
                        subscribe_msg.append({
                            "type": stype,
                            "codes": markets,
                        })
                    subscribe_msg.append({"format": "DEFAULT"})

                    await ws.send(json.dumps(subscribe_msg))

                    # 메시지 수신 루프
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            if isinstance(raw_msg, bytes):
                                data = json.loads(raw_msg.decode("utf-8"))
                            else:
                                data = json.loads(raw_msg)

                            msg_type = data.get("type", "")
                            for cb in self._callbacks.get(msg_type, []):
                                try:
                                    await cb(data)
                                except Exception as e:
                                    logger.error(f"콜백 처리 오류 ({msg_type}): {e}")
                        except json.JSONDecodeError:
                            logger.warning("WebSocket 메시지 파싱 실패")

            except websockets.ConnectionClosed:
                if not self._running:
                    break
                logger.warning(f"WebSocket 연결 끊김, {self._reconnect_delay}초 후 재연결...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"WebSocket 오류: {e}, {self._reconnect_delay}초 후 재연결...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    async def close(self) -> None:
        """WebSocket 종료."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("WebSocket 연결 종료")
