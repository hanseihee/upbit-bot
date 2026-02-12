"""전략 파라미터, API 설정 등 모든 설정값을 dataclass로 관리."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


@dataclass
class TradingConfig:
    """전체 트레이딩 봇 설정."""

    # ── API ──────────────────────────────────────────────
    api_url: str = "https://api.upbit.com/v1"
    ws_url: str = "wss://api.upbit.com/websocket/v1"
    access_key: str = field(default_factory=lambda: os.getenv("UPBIT_ACCESS_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("UPBIT_SECRET_KEY", ""))

    # ── 전략 파라미터 ───────────────────────────────────
    grid_levels: int = 4
    grid_spacing_atr_mult: float = 5.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20
    adx_period: int = 14
    regime_bb_width_low: float = 0.005   # BB Width < 0.5% → 횡보
    regime_bb_width_high: float = 0.02   # BB Width > 2% → 강한 추세
    regime_adx_threshold: float = 25.0   # ADX > 25 → 추세

    # ── 코인 선별 ───────────────────────────────────────
    max_coins: int = 1
    fixed_coins: list[str] = field(default_factory=lambda: [
        "KRW-BTC"
    ])
    min_volume_krw: float = 1_000_000_000  # 24h 거래대금 10억 이상
    max_daily_change: float = 0.10          # 일 변동률 10% 초과 제외
    max_spread_pct: float = 0.003           # 스프레드 0.3% 이하

    # ── 리스크 관리 ─────────────────────────────────────
    max_per_coin_ratio: float = 0.50
    stop_loss_pct: float = 0.015
    max_drawdown_pct: float = 0.10
    max_position_age_hours: int = 2  # 1분봉 기준 2시간

    # ── 그리드 리사이클 ────────────────────────────────
    grid_recycle_enabled: bool = True
    grid_recycle_rsi_threshold: float = 45.0  # 재진입 RSI 기준
    grid_recycle_cooldown_candles: int = 30   # 재진입 쿨다운 (30분)

    # ── 실행 설정 ───────────────────────────────────────
    paper_balance: float = 100_000     # Paper 모드 초기 잔고
    candle_unit: int = 1               # 1분봉
    check_interval_sec: int = 45       # 45초마다 체크
    min_order_krw: float = 5_000       # Upbit 최소 주문금액
    coin_reselect_hours: int = 24      # 24시간마다 코인 재선별
    upbit_fee_rate: float = 0.0005     # 0.05% 수수료

    # ── Rate Limit ──────────────────────────────────────
    market_rate_limit: int = 10        # 시세 API 초당 10회
    order_rate_limit: int = 8          # 주문 API 초당 8회

    # ── 모멘텀 전략 ──────────────────────────────────────
    momentum_enabled: bool = True
    momentum_rsi_threshold: float = 55.0       # RSI 모멘텀 기준
    momentum_volume_mult: float = 1.5          # 거래량 배수 기준
    momentum_trailing_stop_pct: float = 0.008  # 트레일링 스톱 0.8%
    momentum_atr_stop_mult: float = 2.0        # ATR 스톱 배수
    momentum_max_hold_minutes: int = 15        # 최대 보유 시간
    momentum_hard_stop_pct: float = 0.012      # 하드 손절 1.2%
    momentum_position_ratio: float = 0.20      # 가용잔고 대비 20%
    momentum_max_positions: int = 1            # 최대 동시 포지션
    momentum_rsi_exit: float = 72.0            # RSI 청산 기준

    # ── 신호 ────────────────────────────────────────────
    min_signal_confidence: float = 0.6

    # ── 텔레그램 ──────────────────────────────────────────
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    telegram_enabled: bool = field(default_factory=lambda: bool(os.getenv("TELEGRAM_BOT_TOKEN", "")))

    # ── 경로 ────────────────────────────────────────────
    db_path: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent.parent / "data" / "trades.db"
        )
    )
    log_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent.parent / "logs"
        )
    )

    def validate(self) -> list[str]:
        """설정값 유효성 검사. 문제가 있으면 에러 메시지 리스트 반환."""
        errors: list[str] = []
        if not self.access_key:
            errors.append("UPBIT_ACCESS_KEY가 설정되지 않았습니다.")
        if not self.secret_key:
            errors.append("UPBIT_SECRET_KEY가 설정되지 않았습니다.")
        if self.grid_levels < 2:
            errors.append("grid_levels는 최소 2 이상이어야 합니다.")
        if self.min_order_krw < 5000:
            errors.append("Upbit 최소 주문금액은 5,000원입니다.")
        if self.stop_loss_pct <= 0 or self.stop_loss_pct >= 1:
            errors.append("stop_loss_pct는 0~1 사이여야 합니다.")
        return errors
