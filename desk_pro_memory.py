#!/usr/bin/env python3
"""
DESK PRO Adaptive Trading Engine — safety-first foundation.

- Persistent SQLite journal and strategy-version history.
- EMA trend strategy with controlled parameter selection on chronological
  train/validation splits (not random shuffling).
- Multi-market labels: crypto, stocks, and 15-minute prediction simulations.
- Explicit spread, slippage, fee, and delayed-fill assumptions in backtests.
- Risk caps and a circuit breaker.
- Live execution is intentionally NOT implemented or enabled. Connect a
  broker/exchange-specific adapter only after testing and security review.

CSV input format: timestamp,open,high,low,close,volume
Timestamps should be ISO-8601. Prices must be positive.
Python 3.10+, standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

APP_NAME = "DESK PRO Adaptive"
DEFAULT_DB = Path("desk_pro_memory.sqlite3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def valid(self) -> bool:
        vals = (self.open, self.high, self.low, self.close)
        return all(math.isfinite(x) and x > 0 for x in vals) and (
            self.low <= min(self.open, self.close)
            and self.high >= max(self.open, self.close)
            and self.low <= self.high
        )


@dataclass(frozen=True)
class RiskConfig:
    starting_cash: float = 10.0
    max_position_fraction: float = 0.10
    max_trade_risk_fraction: float = 0.01
    max_daily_loss_fraction: float = 0.03
    max_drawdown_fraction: float = 0.10
    max_open_positions: int = 1

    def validate(self) -> None:
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        for name in (
            "max_position_fraction", "max_trade_risk_fraction",
            "max_daily_loss_fraction", "max_drawdown_fraction"
        ):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")


class MemoryStore:
    """Persistent journal; memory is evidence, not an instruction to trade."""

    def __init__(self, path: Path = DEFAULT_DB):
        self.path = path
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            fees REAL NOT NULL,
            pnl REAL,
            strategy_version TEXT NOT NULL,
            reason TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            observation TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS strategy_versions (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            version TEXT NOT NULL,
            params_json TEXT NOT NULL,
            train_score REAL NOT NULL,
            validation_score REAL NOT NULL,
            accepted INTEGER NOT NULL,
            explanation TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        """)
        self.db.commit()

    def event(self, level: str, message: str, metadata: dict | None = None) -> None:
        self.db.execute(
            "INSERT INTO system_events(created_at,level,message,metadata_json) VALUES(?,?,?,?)",
            (utc_now(), level, message, json.dumps(metadata or {}))
        )
        self.db.commit()

    def observation(self, market: str, symbol: str, text: str,
                    metadata: dict | None = None) -> None:
        self.db.execute(
            "INSERT INTO observations(created_at,market,symbol,observation,metadata_json) VALUES(?,?,?,?,?)",
            (utc_now(), market, symbol, text, json.dumps(metadata or {}))
        )
        self.db.commit()

    def journal(self, market: str, symbol: str, action: str, price: float,
                quantity: float, fees: float, pnl: float | None,
                version: str, reason: str, metadata: dict | None = None) -> None:
        self.db.execute(
            """INSERT INTO trade_journal
            (created_at,market,symbol,action,price,quantity,fees,pnl,
             strategy_version,reason,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (utc_now(), market, symbol, action, price, quantity, fees, pnl,
             version, reason, json.dumps(metadata or {}))
        )
        self.db.commit()

    def strategy_version(self, version: str, params: dict, train: float,
                         validation: float, accepted: bool, explanation: str) -> None:
        self.db.execute(
            """INSERT INTO strategy_versions
            (created_at,version,params_json,train_score,validation_score,accepted,explanation)
            VALUES(?,?,?,?,?,?,?)""",
            (utc_now(), version, json.dumps(params), train, validation,
             int(accepted), explanation)
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def load_csv(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "open", "high", "low", "close"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must contain timestamp,open,high,low,close; volume is optional")
        for line, row in enumerate(reader, start=2):
            try:
                c = Candle(
                    timestamp=str(row["timestamp"]),
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row.get("volume") or 0),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line}") from exc
            if not c.valid():
                raise ValueError(f"Invalid OHLC values on CSV line {line}")
            candles.append(c)
    candles.sort(key=lambda c: c.timestamp)
    if len(candles) < 100:
        raise ValueError("Use at least 100 candles for a minimally useful test")
    return candles


def ema(values: list[float], period: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    if period < 2:
        raise ValueError("EMA period must be at least 2")
    alpha = 2.0 / (period + 1)
    current: Optional[float] = None
    for i, value in enumerate(values):
        current = value if current is None else alpha * value + (1 - alpha) * current
        out[i] = current
    return out


@dataclass
class BacktestResult:
    final_equity: float
    return_fraction: float
    max_drawdown: float
    trades: int
    win_rate: float
    fees_paid: float
    score: float


def backtest(candles: list[Candle], fast: int, slow: int, fee_bps: float,
             spread_bps: float, slippage_bps: float, delay_bars: int,
             initial_cash: float = 10.0) -> BacktestResult:
    """Long/flat EMA test. Orders fill after delay_bars at adverse estimated costs."""
    if fast >= slow or fast < 2 or delay_bars < 0:
        raise ValueError("Require fast < slow, fast >= 2, and nonnegative delay")
    closes = [c.close for c in candles]
    ef, es = ema(closes, fast), ema(closes, slow)
    cash, qty, entry_price = initial_cash, 0.0, 0.0
    equity_peak, max_dd, fees_paid = initial_cash, 0.0, 0.0
    closed_pnls: list[float] = []
    pending: list[tuple[int, str]] = []
    cost_rate = (fee_bps + spread_bps / 2 + slippage_bps) / 10000.0

    for i in range(1, len(candles)):
        # Execute due orders at current bar open, never at the signal close.
        due = [p for p in pending if p[0] <= i]
        pending = [p for p in pending if p[0] > i]
        for _, action in due:
            px = candles[i].open
            if action == "BUY" and qty == 0 and cash > 0:
                fill = px * (1 + cost_rate)
                budget = cash * 0.95
                qty = budget / fill
                notional = qty * fill
                fee = notional * fee_bps / 10000
                cash -= notional + fee
                fees_paid += fee
                entry_price = fill
            elif action == "SELL" and qty > 0:
                fill = px * (1 - cost_rate)
                proceeds = qty * fill
                fee = proceeds * fee_bps / 10000
                pnl = proceeds - fee - (qty * entry_price)
                cash += proceeds - fee
                fees_paid += fee
                closed_pnls.append(pnl)
                qty, entry_price = 0.0, 0.0

        # Signal is based only on information available at the previous close.
        if ef[i-1] is not None and es[i-1] is not None and ef[i] is not None and es[i] is not None:
            if ef[i-1] <= es[i-1] and ef[i] > es[i] and qty == 0:
                pending.append((i + delay_bars, "BUY"))
            elif ef[i-1] >= es[i-1] and ef[i] < es[i] and qty > 0:
                pending.append((i + delay_bars, "SELL"))

        equity = cash + qty * candles[i].close
        equity_peak = max(equity_peak, equity)
        max_dd = max(max_dd, (equity_peak - equity) / equity_peak if equity_peak else 0.0)

    if qty > 0:
        px = candles[-1].close * (1 - cost_rate)
        proceeds = qty * px
        fee = proceeds * fee_bps / 10000
        pnl = proceeds - fee - qty * entry_price
        cash += proceeds - fee
        fees_paid += fee
        closed_pnls.append(pnl)
        qty = 0.0

    final_equity = cash
    ret = final_equity / initial_cash - 1
    win_rate = sum(p > 0 for p in closed_pnls) / len(closed_pnls) if closed_pnls else 0.0
    # Penalize drawdown and low sample sizes; this is a ranking score, not a forecast.
    score = ret - 1.5 * max_dd - (0.02 / math.sqrt(len(closed_pnls)) if closed_pnls else 0.02)
    return BacktestResult(final_equity, ret, max_dd, len(closed_pnls), win_rate, fees_paid, score)


def choose_parameters(candles: list[Candle], store: MemoryStore,
                      fee_bps: float, spread_bps: float, slippage_bps: float,
                      delay_bars: int) -> dict:
    """Chronological train/holdout validation; only promote a candidate if it passes gates."""
    split = int(len(candles) * 0.70)
    train, holdout = candles[:split], candles[max(0, split - 50):]
    candidates = []
    for fast in (5, 8, 10, 12, 16):
        for slow in (20, 24, 30, 40, 50):
            if fast >= slow:
                continue
            tr = backtest(train, fast, slow, fee_bps, spread_bps, slippage_bps,
                          delay_bars)
            candidates.append((tr.score, fast, slow, tr))
    candidates.sort(reverse=True, key=lambda x: x[0])
    _, best_fast, best_slow, train_result = candidates[0]
    val = backtest(holdout, best_fast, best_slow, fee_bps, spread_bps,
                   slippage_bps, delay_bars)
    # Require nonnegative holdout return and bounded drawdown before promoting.
    accepted = val.return_fraction > 0 and val.max_drawdown <= 0.10 and val.trades >= 2
    params = {"fast_ema": best_fast, "slow_ema": best_slow,
              "fee_bps": fee_bps, "spread_bps": spread_bps,
              "slippage_bps": slippage_bps, "delay_bars": delay_bars}
    version = f"ema-f{best_fast}-s{best_slow}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    explanation = ("Accepted on holdout gates; still requires paper-forward testing."
                   if accepted else "Rejected: holdout return/trade-count/drawdown gate failed.")
    store.strategy_version(version, params, train_result.score, val.score, accepted, explanation)
    store.event("INFO" if accepted else "WARNING", explanation,
                {"params": params, "validation": asdict(val)})
    return {"version": version, "params": params, "accepted": accepted,
            "train": asdict(train_result), "validation": asdict(val),
            "explanation": explanation}


def risk_check(equity: float, day_start_equity: float, peak_equity: float,
               config: RiskConfig) -> tuple[bool, str]:
    if equity <= 0:
        return False, "Equity is zero or below"
    if equity <= day_start_equity * (1 - config.max_daily_loss_fraction):
        return False, "Daily loss cutoff reached"
    if equity <= peak_equity * (1 - config.max_drawdown_fraction):
        return False, "Maximum drawdown cutoff reached"
    return True, "Risk checks passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--csv", type=Path, help="Historical OHLC CSV file")
    parser.add_argument("--market", choices=("crypto", "stocks", "prediction15m"),
                        default="crypto")
    parser.add_argument("--symbol", default="DEMO")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--spread-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--delay-bars", type=int, default=1)
    parser.add_argument("--live", action="store_true",
                        help="Intentionally blocked; live execution is not implemented.")
    args = parser.parse_args()

    if args.live or os.environ.get("DESK_PRO_LIVE_TRADING") == "1":
        print("BLOCKED: live order execution is not implemented in this build.")
        print("No orders were sent. Do not add API secrets to source code or GitHub.")
        return 2
    if args.fee_bps < 0 or args.spread_bps < 0 or args.slippage_bps < 0:
        parser.error("cost assumptions cannot be negative")
    if args.delay_bars < 0:
        parser.error("--delay-bars cannot be negative")

    store = MemoryStore(args.db)
    try:
        if not args.csv:
            print("DESK PRO memory engine is ready, but no market data was supplied.")
            print("Provide historical candles with --csv data.csv to run a backtest.")
            print("No live market feed or real-money trading is connected.")
            store.event("INFO", "Startup without data; no trading performed",
                        {"market": args.market, "symbol": args.symbol})
            return 0
        candles = load_csv(args.csv)
        store.observation(args.market, args.symbol, "Historical data loaded",
                          {"candles": len(candles), "first": candles[0].timestamp,
                           "last": candles[-1].timestamp})
        result = choose_parameters(candles, store, args.fee_bps, args.spread_bps,
                                   args.slippage_bps, args.delay_bars)
        print(json.dumps(result, indent=2))
        print(f"Persistent journal: {args.db.resolve()}")
        print("Research/backtest only. Results do not predict future profits.")
        return 0
    except Exception as exc:
        store.event("ERROR", str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
