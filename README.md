# DESK PRO Adaptive Trading Research Engine

A Python research/backtesting foundation with persistent SQLite memory, a trade/observation journal, EMA parameter search, chronological holdout validation, estimated fees/spread/slippage, delayed fills, and basic drawdown checks.

## Safety status

**This build does not place live orders.** The `--live` flag and `DESK_PRO_LIVE_TRADING=1` environment setting are deliberately blocked. No broker or exchange is connected. Do not put API keys, passwords, or tokens in this repository. Live trading needs a separately reviewed broker-specific adapter, secure secret storage, and extensive paper-forward testing first.

## Requirements

- Python 3.10 or newer
- Standard library only
- A historical OHLC CSV file with columns `timestamp,open,high,low,close` and optional `volume`

Use data you are permitted to access. Timestamps should be ISO-8601. At least 100 valid candles are required.

## Quick safe smoke test

Run a complete research/backtest flow without needing to create a CSV first:

```bash
python desk_pro_memory.py --demo
```

This uses deterministic **synthetic candles** only. It is useful for checking that the program starts, chooses parameters, prints a result, and writes its local SQLite journal. Synthetic results are not live prices and do not demonstrate a profitable strategy.

## Run a historical backtest

```bash
python desk_pro_memory.py --csv candles.csv --market crypto --symbol BTC-USD
```

Supported market labels are `crypto`, `stocks`, and `prediction15m`. The prediction-market label is only a label in this initial version; contract pricing, settlement, and Kalshi integration are **not** implemented. The program does not fetch live market data.

Optional cost assumptions (basis points):

```bash
python desk_pro_memory.py --csv candles.csv --market stocks --symbol AAPL --fee-bps 10 --spread-bps 5 --slippage-bps 5 --delay-bars 1
```

The default starting equity for backtests is $10 virtual cash. Backtests use a long/flat EMA strategy and estimate transaction costs. Spread and slippage affect fill prices; fees are charged separately to avoid double-counting. These estimates are not guarantees of real-world execution or profit.

## Run tests

```bash
python -m unittest -v
```

GitHub Actions runs the unit tests on pushes and pull requests and checks that live mode remains blocked. Check the repository's **Actions** tab for the actual workflow result; adding the workflow does not itself mean the tests have passed.

## Persistent memory

The local file `desk_pro_memory.sqlite3` stores observations, system events, strategy versions, and backtest metadata. Keep this database private if it contains personal or sensitive notes. Do not commit API credentials or account information.

## Adaptive strategy rules

Candidate EMA pairs are ranked on a chronological training segment. The selected candidate is checked on a later holdout segment and recorded as accepted/rejected based on predefined return, trade-count, and drawdown gates. Passing those gates is not proof of an edge; further out-of-sample testing and paper-forward testing are necessary. The strategy is not automatically authorized to trade real funds.

## Current limitations

- No live execution or broker/exchange adapter.
- No live market-data feed or multi-source price reconciliation.
- No dedicated prediction-market contract payoff/settlement simulator yet.
- Current strategy is long/flat EMA crossover, not a general AI model.
- Fees, spread, slippage, and fill delay are assumptions; real conditions can be worse.
- The risk helper is not a substitute for a complete portfolio/accounting layer.

Never risk money you cannot afford to lose. Automated trading can lose the entire amount allocated to it.
