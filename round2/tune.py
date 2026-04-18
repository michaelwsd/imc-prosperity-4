"""
Hyperparameter tuning for the Round 1 market maker.

ML methodology:
  - Temporal split: days -2, -1 are TRAIN (2-fold CV). Day 0 is held-out TEST.
  - 2-fold CV: each TRAIN day serves as a held-out validation fold.
  - Score: mean PnL across folds (also track min for robustness).
  - Params are tuned per-product in nested CONFIG dict.

Usage:
    python tune.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import itertools
import json
import copy
from typing import Dict, List, Tuple
from datamodel import Listing, OrderDepth, Trade, TradingState, Order, Observation
from backtester import (
    load_price_data, load_trade_data, build_order_depth,
    match_orders, simulate_resting_fills,
)
from trader import Trader


DATA_DIR = "data"
TRAIN_DAYS = ["-1", "0"]
TEST_DAYS = ["1"]


def day_files(day: str) -> Tuple[str, str]:
    return (f"{DATA_DIR}/prices_round_2_day_{day}.csv",
            f"{DATA_DIR}/trades_round_2_day_{day}.csv")


# =====================================================================
# Backtest harness
# =====================================================================
def backtest_day(trader_cls, price_file: str, trade_file: str,
                 pos_limit: int = 80) -> Dict:
    ticks = load_price_data(price_file)
    bot_trades = load_trade_data(trade_file)
    trader = trader_cls()

    positions = {}
    cash = 0
    traderData = ""
    products = set()
    prev_own_trades = {}
    prev_market_trades = {}
    total_trades = 0
    max_abs_pos = {}
    min_running_pnl = 0.0

    for ts in sorted(ticks.keys()):
        rows = ticks[ts]
        order_depths = {}
        listings = {}
        for row in rows:
            p = row['product']
            products.add(p)
            order_depths[p] = build_order_depth(row)
            listings[p] = Listing(p, p, 'XIRECS')
            if p not in positions:
                positions[p] = 0
                max_abs_pos[p] = 0

        market_trades_this_tick = {}
        tick_bot_trades = bot_trades.get(ts, [])
        for bt in tick_bot_trades:
            sym = bt['symbol']
            if sym not in market_trades_this_tick:
                market_trades_this_tick[sym] = []
            market_trades_this_tick[sym].append(
                Trade(sym, int(float(bt['price'])), int(bt['quantity']),
                      buyer=bt.get('buyer', ''), seller=bt.get('seller', ''),
                      timestamp=ts))

        state = TradingState(
            traderData=traderData, timestamp=ts, listings=listings,
            order_depths=order_depths, own_trades=prev_own_trades,
            market_trades={**{p: [] for p in products}, **prev_market_trades},
            position=dict(positions), observations=Observation({}, {}))

        try:
            result, conversions, traderData = trader.run(state)
        except Exception as e:
            return {"pnl": -1e9, "positions_final": dict(positions),
                    "trades": total_trades, "max_abs_pos": max_abs_pos,
                    "drawdown": -1e9, "error": str(e)}

        tick_own_trades = {}
        for p in products:
            orders = result.get(p, [])
            od = None
            for row in rows:
                if row['product'] == p:
                    od = build_order_depth(row)
                    break
            if od is None:
                continue
            trades, np2, pnl, resting = match_orders(
                orders, od, p, positions[p], pos_limit)
            resting_trades, np2, resting_pnl = simulate_resting_fills(
                resting, tick_bot_trades, p, np2, pos_limit)
            trades.extend(resting_trades)
            pnl += resting_pnl
            positions[p] = np2
            cash += pnl
            total_trades += len(trades)
            tick_own_trades[p] = trades
            max_abs_pos[p] = max(max_abs_pos[p], abs(np2))

        prev_own_trades = tick_own_trades
        prev_market_trades = market_trades_this_tick

        mid_prices = {}
        for row in rows:
            mp = float(row['mid_price'])
            if mp > 0:
                mid_prices[row['product']] = mp
        mtm = sum(positions.get(p, 0) * mid_prices.get(p, 0) for p in mid_prices)
        min_running_pnl = min(min_running_pnl, cash + mtm)

    last_ts = max(ticks.keys())
    mid_prices = {}
    for row in ticks[last_ts]:
        mp = float(row['mid_price'])
        if mp > 0:
            mid_prices[row['product']] = mp
    unrealized = sum(positions.get(p, 0) * mid_prices.get(p, 0) for p in mid_prices)
    total_pnl = cash + unrealized

    return {
        "pnl": total_pnl,
        "positions_final": dict(positions),
        "trades": total_trades,
        "max_abs_pos": max_abs_pos,
        "drawdown": min_running_pnl,
    }


# =====================================================================
# Build a Trader with a modified CONFIG
# =====================================================================
def make_trader(config_overrides: Dict[str, Dict]) -> type:
    """
    config_overrides: {"PRODUCT_NAME": {"field": value, ...}}
    Creates a Trader subclass with CONFIG deep-merged with overrides.
    """
    new_config = copy.deepcopy(Trader.CONFIG)
    for product, overrides in config_overrides.items():
        new_config[product].update(overrides)
    return type("TunedTrader", (Trader,), {"CONFIG": new_config})


def cv_score(config_overrides: Dict, days: List[str]) -> Dict:
    cls = make_trader(config_overrides)
    pnls = []
    for day in days:
        pf, tf = day_files(day)
        res = backtest_day(cls, pf, tf)
        pnls.append(res["pnl"])
    return {"pnls": pnls, "mean": sum(pnls)/len(pnls), "min": min(pnls)}


# =====================================================================
# Per-product grid search (only one product tuned at a time)
# =====================================================================
def grid_search_product(product: str, grid: Dict[str, list],
                        days: List[str], locked: Dict[str, Dict]):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    results = []
    total = len(combos)
    print(f"  Grid: {total} combos × {len(days)} folds = {total*len(days)} runs")
    for i, combo in enumerate(combos):
        overrides = {product: dict(zip(keys, combo))}
        # Merge locked overrides for OTHER products
        for p, pcfg in locked.items():
            if p != product:
                overrides[p] = pcfg
        scores = cv_score(overrides, days)
        results.append((dict(zip(keys, combo)), scores))
        if (i + 1) % max(1, total // 10) == 0 or i == total - 1:
            best = max(r[1]['mean'] for r in results)
            print(f"  [{i+1}/{total}] best_so_far={best:>8,.0f}")
    results.sort(key=lambda r: r[1]["mean"], reverse=True)
    return results


def print_top(results, label, n=5):
    print(f"\n  TOP {n} {label}:")
    for i, (p, s) in enumerate(results[:n]):
        print(f"  #{i+1}: mean={s['mean']:>7,.0f}  min={s['min']:>7,.0f}")
        for k, v in p.items():
            print(f"      {k} = {v}")


# =====================================================================
# Main tuning sequence
# =====================================================================
if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#" + " Round 2 Market Maker — CV Parameter Tuning ".center(68) + "#")
    print("#"*70)
    print(f" Train: {TRAIN_DAYS}   Test: {TEST_DAYS}")

    # ── Baseline ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(" BASELINE (current trader.py CONFIG)")
    print("="*70)
    for day in TRAIN_DAYS + TEST_DAYS:
        pf, tf = day_files(day)
        res = backtest_day(Trader, pf, tf)
        label = "TRAIN" if day in TRAIN_DAYS else "TEST "
        print(f"  Day {day:>2} ({label}): PnL={res['pnl']:>9,.0f}  "
              f"max_pos={res['max_abs_pos']}  final={res['positions_final']}")

    # ── Tune OSMIUM ────────────────────────────────────────────────
    print("\n" + "="*70)
    print(" TUNING: OSMIUM")
    print("="*70)
    osm_grid = {
        "ema_span":      [10, 15, 20, 30],
        "take_edge":     [0, 1],
        "skew_coef":     [0.03, 0.04, 0.05, 0.06, 0.08],
        "layers":        [
            [(i, max(4, i)) for i in range(1, 13)],
            [(i, 8) for i in range(1, 13)],
            [(i, 6) for i in range(1, 13)],
            [(i, max(3, i-1)) for i in range(1, 13)],
        ],
    }
    osm_results = grid_search_product(Trader.OSM, osm_grid, TRAIN_DAYS, locked={})
    print_top(osm_results, "OSMIUM")
    best_osm = osm_results[0][0]

    # ── Tune PEPPER (with OSMIUM locked) ───────────────────────────
    print("\n" + "="*70)
    print(" TUNING: PEPPER")
    print("="*70)
    pep_grid = {
        "ema_span":       [5, 8, 12, 15],
        "drift_horizon":  [50, 60, 70, 85, 100],
        "take_edge":      [0, 1],
        "skew_coef":      [0.03, 0.04, 0.05, 0.06],
        "layers":         [
            [(i, max(5, i+3)) for i in range(1, 11)],
            [(i, 8) for i in range(1, 11)],
            [(i, 10) for i in range(1, 11)],
            [(i, max(4, i+2)) for i in range(1, 11)],
        ],
    }
    pep_results = grid_search_product(
        Trader.PEP, pep_grid, TRAIN_DAYS,
        locked={Trader.OSM: best_osm})
    print_top(pep_results, "PEPPER")
    best_pep = pep_results[0][0]

    # ── Final evaluation on test set ───────────────────────────────
    print("\n" + "="*70)
    print(" BEST COMBINED CONFIG")
    print("="*70)
    print(f"  OSMIUM: {best_osm}")
    print(f"  PEPPER: {best_pep}")

    print("\n" + "="*70)
    print(" FINAL EVALUATION")
    print("="*70)
    best_cls = make_trader({Trader.OSM: best_osm, Trader.PEP: best_pep})
    total_train = 0
    total_test = 0
    for day in TRAIN_DAYS:
        pf, tf = day_files(day)
        res = backtest_day(best_cls, pf, tf)
        total_train += res["pnl"]
        print(f"  Day {day} (TRAIN): PnL={res['pnl']:>9,.0f}  max_pos={res['max_abs_pos']}")
    for day in TEST_DAYS:
        pf, tf = day_files(day)
        res = backtest_day(best_cls, pf, tf)
        total_test += res["pnl"]
        print(f"  Day {day} (TEST ): PnL={res['pnl']:>9,.0f}  max_pos={res['max_abs_pos']}")
    print(f"  {'─'*60}")
    print(f"  Train total: {total_train:>10,.0f}")
    print(f"  Test  total: {total_test:>10,.0f}")
    print(f"  Grand total: {total_train + total_test:>10,.0f}")

    # Save best config
    out = {Trader.OSM: best_osm, Trader.PEP: best_pep}
    with open("best_params.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved to best_params.json")
