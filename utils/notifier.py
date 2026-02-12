"""콘솔 알림 모듈. 텔레그램 확장 가능."""

from __future__ import annotations

from loguru import logger


class Notifier:
    """알림 전송 (현재 콘솔 로깅만 구현)."""

    def notify(self, title: str, message: str, level: str = "info") -> None:
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"[알림] {title}: {message}")

    def trade_executed(self, side: str, market: str, price: float, volume: float) -> None:
        emoji = "🟢" if side == "bid" else "🔴"
        action = "매수" if side == "bid" else "매도"
        self.notify(
            f"{emoji} {action} 체결",
            f"{market} | 가격: {price:,.0f} | 수량: {volume:.8f}",
        )

    def stop_loss_triggered(self, market: str, loss_pct: float) -> None:
        self.notify(
            "🚨 손절 발동",
            f"{market} | 손실: {loss_pct:.2%}",
            level="warning",
        )

    def drawdown_alert(self, current_drawdown: float, threshold: float) -> None:
        self.notify(
            "🚨 드로다운 경고",
            f"현재 드로다운: {current_drawdown:.2%} / 한도: {threshold:.2%}",
            level="error",
        )

    def grid_created(self, market: str, levels: int, base_price: float) -> None:
        self.notify(
            "📊 그리드 생성",
            f"{market} | {levels}단계 | 기준가: {base_price:,.0f}",
        )
