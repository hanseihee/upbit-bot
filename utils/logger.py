"""loguru 기반 로깅 시스템. 콘솔 + 파일 출력, 거래 로그 별도 관리."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_configured = False


def setup_logger(log_dir: str | None = None) -> None:
    """로거 초기화. 중복 호출 시 무시."""
    global _configured
    if _configured:
        return
    _configured = True

    log_path = Path(log_dir) if log_dir else Path(__file__).resolve().parent.parent / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    # 기본 핸들러 제거 후 재설정
    logger.remove()

    # 콘솔 출력
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        colorize=True,
    )

    # 전체 로그 파일 (DEBUG 이상)
    logger.add(
        str(log_path / "bot_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}",
    )

    # 거래 전용 로그 (INFO 이상, trade 태그만)
    logger.add(
        str(log_path / "trades_{time:YYYY-MM-DD}.log"),
        level="INFO",
        rotation="00:00",
        retention="90 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        filter=lambda record: "trade" in record["extra"],
    )


def get_trade_logger():
    """거래 전용 로거 반환."""
    return logger.bind(trade=True)
