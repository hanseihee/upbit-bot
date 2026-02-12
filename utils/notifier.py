"""콘솔 + 텔레그램 알림 모듈."""

from __future__ import annotations

import asyncio
from datetime import datetime

import httpx
from loguru import logger


class Notifier:
    """알림 전송 (콘솔 로깅 + 텔레그램)."""

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        enabled: bool = False,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = enabled and bool(bot_token) and bool(chat_id)
        self._base_url = f"https://api.telegram.org/bot{bot_token}"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _send_telegram(self, text: str) -> None:
        """텔레그램 메시지 전송."""
        if not self._enabled:
            return
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            if resp.status_code != 200:
                logger.warning(f"텔레그램 전송 실패: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"텔레그램 전송 오류: {e}")

    def _send_async(self, text: str) -> None:
        """동기 컨텍스트에서 비동기 전송 스케줄링."""
        if not self._enabled:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send_telegram(text))
        except RuntimeError:
            pass

    # ── 공통 알림 ─────────────────────────────────────

    def notify(self, title: str, message: str, level: str = "info") -> None:
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"[알림] {title}: {message}")
        self._send_async(f"<b>{title}</b>\n{message}")

    # ── 거래 알림 ─────────────────────────────────────

    def trade_executed(self, side: str, market: str, price: float, volume: float) -> None:
        emoji = "🟢" if side == "bid" else "🔴"
        action = "매수" if side == "bid" else "매도"
        title = f"{emoji} {action} 체결"
        body = f"{market} | 가격: {price:,.0f} | 수량: {volume:.8f}"
        logger.info(f"[알림] {title}: {body}")
        self._send_async(
            f"<b>{title}</b>\n"
            f"마켓: {market}\n"
            f"가격: {price:,.0f} KRW\n"
            f"수량: {volume:.8f}"
        )

    def stop_loss_triggered(self, market: str, loss_pct: float) -> None:
        title = "🚨 손절 발동"
        body = f"{market} | 손실: {loss_pct:.2%}"
        logger.warning(f"[알림] {title}: {body}")
        self._send_async(f"<b>{title}</b>\n마켓: {market}\n손실: {loss_pct:.2%}")

    def drawdown_alert(self, current_drawdown: float, threshold: float) -> None:
        title = "🚨 드로다운 경고"
        body = f"현재: {current_drawdown:.2%} / 한도: {threshold:.2%}"
        logger.error(f"[알림] {title}: {body}")
        self._send_async(f"<b>{title}</b>\n현재: {current_drawdown:.2%}\n한도: {threshold:.2%}")

    def grid_created(self, market: str, levels: int, base_price: float) -> None:
        title = "📊 그리드 생성"
        body = f"{market} | {levels}단계 | 기준가: {base_price:,.0f}"
        logger.info(f"[알림] {title}: {body}")
        self._send_async(
            f"<b>{title}</b>\n"
            f"마켓: {market}\n"
            f"레벨: {levels}단계\n"
            f"기준가: {base_price:,.0f} KRW"
        )

    # ── 모멘텀 알림 ───────────────────────────────────

    def momentum_entry(self, market: str, price: float, order_krw: float) -> None:
        title = "⚡ 모멘텀 진입"
        logger.info(f"[알림] {title}: {market} | {price:,.0f} | {order_krw:,.0f}원")
        self._send_async(
            f"<b>{title}</b>\n"
            f"마켓: {market}\n"
            f"가격: {price:,.0f} KRW\n"
            f"금액: {order_krw:,.0f} KRW"
        )

    def momentum_exit(self, market: str, price: float, pnl_krw: float, pnl_pct: float, reason: str) -> None:
        emoji = "💰" if pnl_krw >= 0 else "💸"
        title = f"{emoji} 모멘텀 청산"
        logger.info(f"[알림] {title}: {market} | {pnl_krw:+,.0f}원 ({pnl_pct:+.2%}) | {reason}")
        self._send_async(
            f"<b>{title}</b>\n"
            f"마켓: {market}\n"
            f"가격: {price:,.0f} KRW\n"
            f"손익: {pnl_krw:+,.0f} KRW ({pnl_pct:+.2%})\n"
            f"사유: {reason}"
        )

    # ── 상태 알림 ─────────────────────────────────────

    def bot_started(self, mode: str, balance: float, coins: list[str]) -> None:
        title = "🤖 봇 시작"
        logger.info(f"[알림] {title}: {mode} | {balance:,.0f}원 | {coins}")
        self._send_async(
            f"<b>{title}</b>\n"
            f"모드: {mode}\n"
            f"잔고: {balance:,.0f} KRW\n"
            f"코인: {', '.join(coins)}"
        )

    def bot_stopped(self, reason: str = "수동 종료") -> None:
        title = "⛔ 봇 중지"
        logger.info(f"[알림] {title}: {reason}")
        self._send_async(f"<b>{title}</b>\n사유: {reason}")

    def daily_summary(self, pnl: float, total_trades: int, balance: float) -> None:
        emoji = "📈" if pnl >= 0 else "📉"
        title = f"{emoji} 일일 요약"
        logger.info(f"[알림] {title}: 손익 {pnl:+,.0f}원 | 거래 {total_trades}건 | 잔고 {balance:,.0f}원")
        self._send_async(
            f"<b>{title}</b>\n"
            f"일일 손익: {pnl:+,.0f} KRW\n"
            f"거래 횟수: {total_trades}건\n"
            f"현재 잔고: {balance:,.0f} KRW\n"
            f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
