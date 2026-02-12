"""그리드 로직 테스트."""

import pytest

from strategy.grid import AdaptiveGrid, Grid, GridLevel, GridSide, GridStatus


class TestAdaptiveGrid:
    def setup_method(self):
        self.grid_engine = AdaptiveGrid(
            atr_multiplier=0.5,
            stop_loss_pct=0.05,
            fee_rate=0.0005,
        )

    def test_grid_levels_count(self):
        """지정한 수만큼 매수/매도 레벨이 생성되어야 한다."""
        grid = self.grid_engine.calculate_grid("KRW-BTC", 90_000_000, 2_000_000, levels=5)
        assert len(grid.buy_levels) == 5
        assert len(grid.sell_levels) == 5

    def test_buy_levels_below_price(self):
        """매수 레벨은 모두 현재가 아래여야 한다."""
        price = 90_000_000
        grid = self.grid_engine.calculate_grid("KRW-BTC", price, 2_000_000, levels=5)
        for level in grid.buy_levels:
            assert level.price < price

    def test_sell_levels_above_buy_levels(self):
        """각 매도 레벨은 대응 매수 레벨보다 높아야 한다."""
        grid = self.grid_engine.calculate_grid("KRW-BTC", 90_000_000, 2_000_000, levels=5)
        for buy_level in grid.buy_levels:
            sell_level = next(
                (s for s in grid.sell_levels if s.level == buy_level.level), None
            )
            assert sell_level is not None
            assert sell_level.price > buy_level.price

    def test_stop_loss_below_lowest_buy(self):
        """손절가는 최하단 매수가 아래여야 한다."""
        grid = self.grid_engine.calculate_grid("KRW-BTC", 90_000_000, 2_000_000, levels=5)
        lowest_buy = min(l.price for l in grid.buy_levels)
        assert grid.stop_loss_price < lowest_buy

    def test_grid_spacing_minimum(self):
        """그리드 간격은 수수료 × 3 이상이어야 한다."""
        price = 90_000_000
        atr = 100  # 매우 작은 ATR
        grid = self.grid_engine.calculate_grid("KRW-BTC", price, atr, levels=5)

        min_spacing = price * 0.0005 * 3
        assert grid.grid_spacing >= min_spacing

    def test_grid_serialization(self):
        """그리드 JSON 직렬화/역직렬화."""
        grid = self.grid_engine.calculate_grid("KRW-BTC", 90_000_000, 2_000_000, levels=5)
        json_str = grid.to_json()
        restored = Grid.from_json(json_str)

        assert restored.market == grid.market
        assert restored.base_price == grid.base_price
        assert len(restored.levels) == len(grid.levels)

    def test_should_update_grid(self):
        """ATR 변화 20% 이상 시 재조정 필요."""
        assert self.grid_engine.should_update_grid(120, 100, 0.2) is True
        assert self.grid_engine.should_update_grid(110, 100, 0.2) is False
        assert self.grid_engine.should_update_grid(80, 100, 0.2) is True


class TestRoundToTick:
    def test_high_price_tick(self):
        """200만원 이상: 1,000원 단위."""
        result = AdaptiveGrid._round_to_tick(90_123_456, 90_000_000)
        assert result % 1000 == 0

    def test_mid_price_tick(self):
        """100만원~200만원: 500원 단위."""
        result = AdaptiveGrid._round_to_tick(1_234_567, 1_500_000)
        assert result % 500 == 0

    def test_low_price_tick(self):
        """1만원~10만원: 10원 단위."""
        result = AdaptiveGrid._round_to_tick(56_789, 50_000)
        assert result % 10 == 0
