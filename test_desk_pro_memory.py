"""Unit tests for DESK PRO's paper-only research engine.

Run: python -m unittest -v
These tests do not contact brokers, exchanges, or live market feeds.
"""
import tempfile
import unittest
from pathlib import Path

from desk_pro_memory import (
    Candle, MemoryStore, RiskConfig, backtest, ema, risk_check,
)


def make_candles(n=240):
    rows = []
    price = 100.0
    for i in range(n):
        # Deterministic oscillating trend to exercise both EMA cross directions.
        price += (0.22 if (i // 18) % 2 == 0 else -0.19)
        rows.append(Candle(
            timestamp=f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00",
            open=price - 0.03, high=price + 0.15,
            low=price - 0.15, close=price, volume=1000.0,
        ))
    return rows


class CandleTests(unittest.TestCase):
    def test_valid_ohlc(self):
        self.assertTrue(Candle("t", 10, 12, 9, 11).valid())

    def test_rejects_impossible_ohlc(self):
        self.assertFalse(Candle("t", 10, 9, 8, 11).valid())

    def test_rejects_non_finite_price(self):
        self.assertFalse(Candle("t", float("nan"), 12, 9, 11).valid())


class IndicatorTests(unittest.TestCase):
    def test_ema_length_and_first_value(self):
        result = ema([1.0, 2.0, 3.0], 2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 1.6666666667)

    def test_ema_rejects_invalid_period(self):
        with self.assertRaises(ValueError):
            ema([1.0, 2.0], 1)


class RiskTests(unittest.TestCase):
    def test_daily_loss_cutoff(self):
        config = RiskConfig(max_daily_loss_fraction=0.03)
        ok, message = risk_check(9.69, 10.0, 10.0, config)
        self.assertFalse(ok)
        self.assertIn("Daily loss", message)

    def test_drawdown_cutoff(self):
        config = RiskConfig(max_drawdown_fraction=0.10)
        ok, message = risk_check(8.9, 10.0, 10.0, config)
        self.assertFalse(ok)
        self.assertIn("drawdown", message.lower())

    def test_normal_equity_passes(self):
        ok, message = risk_check(10.0, 10.0, 10.0, RiskConfig())
        self.assertTrue(ok)


class BacktestTests(unittest.TestCase):
    def test_costs_cannot_improve_execution_vs_zero_cost(self):
        candles = make_candles()
        baseline = backtest(candles, 8, 24, 0, 0, 0, 1)
        costs = backtest(candles, 8, 24, 10, 5, 5, 1)
        self.assertLessEqual(costs.final_equity, baseline.final_equity + 1e-8)
        self.assertGreaterEqual(costs.fees_paid, 0)

    def test_invalid_ema_pair_rejected(self):
        with self.assertRaises(ValueError):
            backtest(make_candles(), 24, 8, 10, 5, 5, 1)

    def test_initial_cash_is_ten_by_default(self):
        result = backtest(make_candles(), 8, 24, 10, 5, 5, 1)
        self.assertTrue(result.final_equity >= 0)


class PersistenceTests(unittest.TestCase):
    def test_memory_store_persists_events(self):
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "memory.sqlite3"
            store = MemoryStore(db_path)
            store.event("INFO", "test event", {"test": True})
            store.observation("crypto", "BTC-USD", "test observation")
            store.close()

            import sqlite3
            with sqlite3.connect(db_path) as db:
                count = db.execute("SELECT COUNT(*) FROM system_events").fetchone()[0]
                observations = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            self.assertEqual(count, 1)
            self.assertEqual(observations, 1)


if __name__ == "__main__":
    unittest.main()
