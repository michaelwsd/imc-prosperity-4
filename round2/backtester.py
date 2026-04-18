"""
Local Backtester for Round 2.

Usage:
    cd round1 && python backtester.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import math
import random
from datamodel import Listing, OrderDepth, Trade, TradingState, Order, Observation
from trader import Trader


def load_price_data(filepath):
    ticks = {}
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            ts = int(row['timestamp'])
            if ts not in ticks:
                ticks[ts] = []
            ticks[ts].append(row)
    return ticks


def load_trade_data(filepath):
    trades = {}
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            ts = int(row['timestamp'])
            if ts not in trades:
                trades[ts] = []
            trades[ts].append(row)
    return trades


def build_order_depth(row):
    od = OrderDepth()
    for i in range(1, 4):
        pk, vk = f'bid_price_{i}', f'bid_volume_{i}'
        if row.get(pk) and row[pk] != '':
            od.buy_orders[int(row[pk])] = int(row[vk])
    for i in range(1, 4):
        pk, vk = f'ask_price_{i}', f'ask_volume_{i}'
        if row.get(pk) and row[pk] != '':
            od.sell_orders[int(row[pk])] = -int(row[vk])
    return od


def match_orders(orders, order_depth, product, position, pos_limit):
    if not orders:
        return [], position, 0, []
    total_buy = sum(o.quantity for o in orders if o.quantity > 0)
    total_sell = sum(-o.quantity for o in orders if o.quantity < 0)
    if position + total_buy > pos_limit or position - total_sell < -pos_limit:
        return [], position, 0, []
    trades = []
    resting = []
    pnl = 0
    for order in orders:
        if order.quantity > 0:
            remaining = order.quantity
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= order.price and remaining > 0:
                    available = -order_depth.sell_orders[ask_price]
                    fill_qty = min(remaining, available)
                    if fill_qty > 0:
                        trades.append(Trade(product, ask_price, fill_qty,
                                            buyer="SUBMISSION", seller="", timestamp=0))
                        position += fill_qty
                        pnl -= ask_price * fill_qty
                        remaining -= fill_qty
                        order_depth.sell_orders[ask_price] += fill_qty
                        if order_depth.sell_orders[ask_price] == 0:
                            del order_depth.sell_orders[ask_price]
            if remaining > 0:
                resting.append(Order(product, order.price, remaining))
        elif order.quantity < 0:
            remaining = -order.quantity
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price >= order.price and remaining > 0:
                    available = order_depth.buy_orders[bid_price]
                    fill_qty = min(remaining, available)
                    if fill_qty > 0:
                        trades.append(Trade(product, bid_price, fill_qty,
                                            buyer="", seller="SUBMISSION", timestamp=0))
                        position -= fill_qty
                        pnl += bid_price * fill_qty
                        remaining -= fill_qty
                        order_depth.buy_orders[bid_price] -= fill_qty
                        if order_depth.buy_orders[bid_price] == 0:
                            del order_depth.buy_orders[bid_price]
            if remaining > 0:
                resting.append(Order(product, order.price, -remaining))
    return trades, position, pnl, resting


def simulate_resting_fills(resting_orders, bot_trades, product, position, pos_limit):
    trades = []
    pnl = 0
    FILL_RATE = 0.5
    for order in resting_orders:
        if order.quantity > 0:
            for bt in bot_trades:
                if bt['symbol'] == product:
                    trade_price = int(float(bt['price']))
                    trade_qty = int(bt['quantity'])
                    if trade_price <= order.price:
                        fill = min(int(trade_qty * FILL_RATE), order.quantity,
                                   pos_limit - position)
                        if fill > 0:
                            trades.append(Trade(product, order.price, fill,
                                                buyer="SUBMISSION", seller="", timestamp=0))
                            position += fill
                            pnl -= order.price * fill
                            order = Order(order.symbol, order.price, order.quantity - fill)
        elif order.quantity < 0:
            for bt in bot_trades:
                if bt['symbol'] == product:
                    trade_price = int(float(bt['price']))
                    trade_qty = int(bt['quantity'])
                    if trade_price >= order.price:
                        fill = min(int(trade_qty * FILL_RATE), -order.quantity,
                                   pos_limit + position)
                        if fill > 0:
                            trades.append(Trade(product, order.price, fill,
                                                buyer="", seller="SUBMISSION", timestamp=0))
                            position -= fill
                            pnl += order.price * fill
                            order = Order(order.symbol, order.price, order.quantity + fill)
    return trades, position, pnl


def run_backtest(price_file, trade_file, pos_limit=80):
    ticks = load_price_data(price_file)
    bot_trades = load_trade_data(trade_file)
    trader = Trader()
    positions = {}
    cash = 0
    cash_per_prod = {}
    total_trades = 0
    traderData = ""
    products = set()
    prev_own_trades = {}
    prev_market_trades = {}

    print(f"\n{'='*70}")
    print(f"  BACKTESTING: {price_file}")
    print(f"{'='*70}")

    for ts in sorted(ticks.keys()):
        rows = ticks[ts]
        order_depths = {}
        listings = {}
        for row in rows:
            product = row['product']
            products.add(product)
            order_depths[product] = build_order_depth(row)
            listings[product] = Listing(product, product, "XIRECS")
            if product not in positions:
                positions[product] = 0

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
            print(f"  ERROR at t={ts}: {e}")
            import traceback
            traceback.print_exc()
            break

        tick_own_trades = {}
        for product in products:
            orders = result.get(product, [])
            od = None
            for row in rows:
                if row['product'] == product:
                    od = build_order_depth(row)
                    break
            if od is None:
                continue
            trades, new_pos, pnl, resting = match_orders(
                orders, od, product, positions[product], pos_limit)
            resting_trades, new_pos, resting_pnl = simulate_resting_fills(
                resting, tick_bot_trades, product, new_pos, pos_limit)
            trades.extend(resting_trades)
            pnl += resting_pnl
            positions[product] = new_pos
            cash += pnl
            cash_per_prod[product] = cash_per_prod.get(product, 0) + pnl
            total_trades += len(trades)
            tick_own_trades[product] = trades

        prev_own_trades = tick_own_trades
        prev_market_trades = market_trades_this_tick

        if ts % 100000 == 0 or ts == 0:
            mid_prices = {}
            for row in rows:
                mid_prices[row['product']] = float(row['mid_price'])
            mtm = sum(positions.get(p, 0) * mid_prices.get(p, 0) for p in mid_prices)
            total_pnl = cash + mtm
            print(f"  t={ts:>7} | cash={cash:>10.0f} | pos={dict(positions)} | PnL={total_pnl:>10.0f}")

    last_ts = max(ticks.keys())
    mid_prices = {}
    for row in ticks[last_ts]:
        mid_prices[row['product']] = float(row['mid_price'])
    unrealized = sum(positions.get(p, 0) * mid_prices.get(p, 0) for p in mid_prices)
    total_pnl = cash + unrealized

    print(f"\n  {'─'*60}")
    print(f"  RESULTS")
    print(f"  {'─'*60}")
    print(f"  Total trades: {total_trades}")
    print(f"  Final positions: {dict(positions)}")
    for p in sorted(products):
        mtm_p = positions.get(p, 0) * mid_prices.get(p, 0)
        pnl_p = cash_per_prod.get(p, 0) + mtm_p
        print(f"    {p:<24} cash={cash_per_prod.get(p, 0):>12,.0f}  mtm={mtm_p:>12,.0f}  pnl={pnl_p:>10,.0f}")
    print(f"  Cash:        {cash:>12,.0f} XIRECS")
    print(f"  Unrealized:  {unrealized:>12,.0f} XIRECS")
    print(f"  Total PnL:   {total_pnl:>12,.0f} XIRECS")
    print(f"  {'─'*60}")
    return total_pnl


if __name__ == "__main__":
    results = {}
    for day in ["-1", "0", "1"]:
        results[day] = run_backtest(
            f"data/prices_round_2_day_{day}.csv",
            f"data/trades_round_2_day_{day}.csv",
        )
    total = sum(results.values())
    print(f"\n{'='*70}")
    print(f"  COMBINED ROUND 2 RESULTS")
    print(f"{'='*70}")
    for day, pnl in results.items():
        print(f"  Day {day:>3} PnL: {pnl:>12,.0f} XIRECS")
    print(f"  {'─'*40}")
    print(f"  Total PnL:  {total:>12,.0f} XIRECS")
    print(f"{'='*70}")
