"""
Local Backtester for IMC Prosperity 4
======================================
Reads the sample CSV data and simulates the exchange, calling your
Trader.run() on every tick exactly like the real platform does.

Simulates two types of fills:
  1. Aggressive fills — your order crosses a bot's order (immediate)
  2. Resting fills — your order sits in the book, and a bot trades
     against it (simulated using the trades CSV data)

Usage:
    python backtester.py
"""

import csv
import math
import random
from datamodel import Listing, OrderDepth, Trade, TradingState, Order, Observation
from trader import Trader


def load_price_data(filepath: str) -> dict:
    """Load the prices CSV into rows grouped by timestamp."""
    ticks = {}
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            ts = int(row['timestamp'])
            if ts not in ticks:
                ticks[ts] = []
            ticks[ts].append(row)
    return ticks


def load_trade_data(filepath: str) -> dict:
    """Load the trades CSV into trades grouped by timestamp."""
    trades = {}
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            ts = int(row['timestamp'])
            if ts not in trades:
                trades[ts] = []
            trades[ts].append(row)
    return trades


def build_order_depth(row: dict) -> OrderDepth:
    """Convert a CSV row into an OrderDepth object."""
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


def match_orders(orders: list, order_depth: OrderDepth, product: str,
                 position: int, pos_limit: int) -> tuple:
    """
    Simulate the exchange matching engine for aggressive orders.
    Returns (trades, new_position, pnl_change, resting_orders).

    Resting orders are orders that didn't cross the book — they'll be
    checked against bot trade data separately.
    """
    if not orders:
        return [], position, 0, []

    # Position limit check
    total_buy = sum(o.quantity for o in orders if o.quantity > 0)
    total_sell = sum(-o.quantity for o in orders if o.quantity < 0)

    if position + total_buy > pos_limit or position - total_sell < -pos_limit:
        return [], position, 0, []

    trades = []
    resting = []
    pnl = 0

    for order in orders:
        if order.quantity > 0:
            # BUY — match against asks
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
            # SELL — match against bids
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


def simulate_resting_fills(resting_orders: list, bot_trades: list,
                           product: str, position: int, pos_limit: int) -> tuple:
    """
    Simulate bots trading against our resting orders.

    We look at the bot trades that happened this tick. If a bot trade
    occurred at a price where our resting order would be attractive,
    we assume some portion of that trade volume fills our order instead.

    For a resting BUY at price P: if bots traded at price <= P,
    they might sell to us instead (our price is at least as good).

    For a resting SELL at price P: if bots traded at price >= P,
    they might buy from us instead.

    We fill up to 50% of the bot trade volume (conservative estimate —
    we're competing with other bots for the same liquidity).
    """
    trades = []
    pnl = 0
    FILL_RATE = 0.5  # assume we capture half the available volume

    for order in resting_orders:
        if order.quantity > 0:
            # Resting BUY — look for bot trades at or below our price
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
                            order = Order(order.symbol, order.price,
                                          order.quantity - fill)

        elif order.quantity < 0:
            # Resting SELL — look for bot trades at or above our price
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
                            order = Order(order.symbol, order.price,
                                          order.quantity + fill)

    return trades, position, pnl


def run_backtest(price_file: str, trade_file: str, pos_limit: int = 80):
    """Run the full backtest on one day of data."""
    ticks = load_price_data(price_file)
    bot_trades = load_trade_data(trade_file)
    trader = Trader()

    positions = {}
    cash = 0
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

        # Build order depths and listings
        order_depths = {}
        listings = {}
        for row in rows:
            product = row['product']
            products.add(product)
            order_depths[product] = build_order_depth(row)
            listings[product] = Listing(product, product, "XIRECS")
            if product not in positions:
                positions[product] = 0

        # Build market trades for this tick (from bot trade data)
        market_trades_this_tick = {}
        tick_bot_trades = bot_trades.get(ts, [])
        for bt in tick_bot_trades:
            sym = bt['symbol']
            if sym not in market_trades_this_tick:
                market_trades_this_tick[sym] = []
            market_trades_this_tick[sym].append(
                Trade(sym, int(float(bt['price'])), int(bt['quantity']),
                      buyer=bt.get('buyer', ''), seller=bt.get('seller', ''),
                      timestamp=ts)
            )

        # Build TradingState
        state = TradingState(
            traderData=traderData,
            timestamp=ts,
            listings=listings,
            order_depths=order_depths,
            own_trades=prev_own_trades,
            market_trades={**{p: [] for p in products}, **prev_market_trades},
            position=dict(positions),
            observations=Observation({}, {}),
        )

        # Call trader
        try:
            result, conversions, traderData = trader.run(state)
        except Exception as e:
            print(f"  ERROR at t={ts}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Match orders
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

            # Phase 1: aggressive fills
            trades, new_pos, pnl, resting = match_orders(
                orders, od, product, positions[product], pos_limit
            )

            # Phase 2: simulate resting fills using bot trade data
            resting_trades, new_pos, resting_pnl = simulate_resting_fills(
                resting, tick_bot_trades, product, new_pos, pos_limit
            )

            trades.extend(resting_trades)
            pnl += resting_pnl
            positions[product] = new_pos
            cash += pnl
            total_trades += len(trades)
            tick_own_trades[product] = trades

        prev_own_trades = tick_own_trades
        prev_market_trades = market_trades_this_tick

        # Progress reporting
        if ts % 100000 == 0 or ts == 0:
            mid_prices = {}
            for row in rows:
                mid_prices[row['product']] = float(row['mid_price'])
            mtm = sum(positions.get(p, 0) * mid_prices.get(p, 0) for p in mid_prices)
            total_pnl = cash + mtm
            pos_str = {k: v for k, v in positions.items()}
            print(f"  t={ts:>7} | cash={cash:>10.0f} | pos={pos_str} | PnL={total_pnl:>10.0f}")

    # Final summary
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
    print(f"  Cash:        {cash:>12,.0f} XIRECS")
    print(f"  Unrealized:  {unrealized:>12,.0f} XIRECS")
    print(f"  Total PnL:   {total_pnl:>12,.0f} XIRECS")
    print(f"  {'─'*60}")
    return total_pnl


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              IMC PROSPERITY 4 — LOCAL BACKTESTER                   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    pnl_day2 = run_backtest(
        "data/tutorial/prices_round_0_day_-2.csv",
        "data/tutorial/trades_round_0_day_-2.csv"
    )
    pnl_day1 = run_backtest(
        "data/tutorial/prices_round_0_day_-1.csv",
        "data/tutorial/trades_round_0_day_-1.csv"
    )

    print(f"\n{'='*70}")
    print(f"  COMBINED RESULTS")
    print(f"{'='*70}")
    print(f"  Day -2 PnL: {pnl_day2:>10,.0f} XIRECS")
    print(f"  Day -1 PnL: {pnl_day1:>10,.0f} XIRECS")
    print(f"  Total PnL:  {pnl_day2 + pnl_day1:>10,.0f} XIRECS")
    print(f"{'='*70}")
