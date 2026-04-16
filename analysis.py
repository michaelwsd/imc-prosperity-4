"""
Analysis script for IMC Prosperity 4 tutorial data.

Loads price and trade CSVs, then runs a series of analyses to understand
the market microstructure and inform trading strategy decisions.

Usage:
    python analysis.py
"""

import csv
import numpy as np
from collections import Counter


# =====================================================================
#  STEP 1: Load the raw CSV data into Python lists
# =====================================================================
# Each prices CSV has one row per (timestamp, product) with the order
# book snapshot: best 3 bids, best 3 asks, and the mid price.
#
# Each trades CSV has one row per bot-to-bot trade that occurred.

def load_prices(filepath):
    """Read a prices CSV and return a list of row dicts."""
    rows = []
    with open(filepath) as f:
        # csv.DictReader reads each row as a dictionary keyed by column name
        # delimiter=';' because these files use semicolons, not commas
        for row in csv.DictReader(f, delimiter=';'):
            rows.append(row)
    return rows


def load_trades(filepath):
    """Read a trades CSV and return a list of row dicts."""
    rows = []
    with open(filepath) as f:
        for row in csv.DictReader(f, delimiter=';'):
            rows.append(row)
    return rows


# Load both days of data
prices_d2 = load_prices("data/tutorial/prices_round_0_day_-2.csv")
prices_d1 = load_prices("data/tutorial/prices_round_0_day_-1.csv")
trades_d2 = load_trades("data/tutorial/trades_round_0_day_-2.csv")
trades_d1 = load_trades("data/tutorial/trades_round_0_day_-1.csv")

print(f"Loaded {len(prices_d2)} price rows (day -2), {len(prices_d1)} (day -1)")
print(f"Loaded {len(trades_d2)} trade rows (day -2), {len(trades_d1)} (day -1)")


# =====================================================================
#  STEP 2: Organize data by product
# =====================================================================
# The raw CSV mixes all products together. We split them into per-product
# arrays so we can analyze each one independently.

def organize_by_product(price_rows):
    """
    Takes raw price rows and returns a dict like:
    {
        "EMERALDS": {
            "timestamps": [0, 100, 200, ...],
            "mids":       [10000.0, 10000.0, ...],
            "spreads":    [16, 16, 8, ...],
            "bid1_vols":  [11, 13, ...],
            "ask1_vols":  [11, 13, ...],
        },
        "TOMATOES": { ... }
    }
    """
    data = {}
    for row in price_rows:
        product = row['product']
        if product not in data:
            data[product] = {
                'timestamps': [], 'mids': [], 'spreads': [],
                'bid1_vols': [], 'ask1_vols': [],
            }

        d = data[product]
        d['timestamps'].append(int(row['timestamp']))
        d['mids'].append(float(row['mid_price']))

        # Calculate spread (ask - bid) if both sides exist
        if row['bid_price_1'] and row['ask_price_1']:
            spread = int(row['ask_price_1']) - int(row['bid_price_1'])
            d['spreads'].append(spread)
            d['bid1_vols'].append(int(row['bid_volume_1']))
            d['ask1_vols'].append(int(row['ask_volume_1']))
        else:
            d['spreads'].append(None)
            d['bid1_vols'].append(None)
            d['ask1_vols'].append(None)

    return data


def organize_trades(trade_rows):
    """
    Takes raw trade rows and returns a dict like:
    {
        "EMERALDS": {
            "timestamps": [900, 1700, ...],
            "prices":     [10008, 9992, ...],
            "quantities": [3, 5, ...],
        }
    }
    """
    data = {}
    for row in trade_rows:
        product = row['symbol']
        if product not in data:
            data[product] = {'timestamps': [], 'prices': [], 'quantities': []}
        d = data[product]
        d['timestamps'].append(int(row['timestamp']))
        d['prices'].append(float(row['price']))
        d['quantities'].append(int(row['quantity']))
    return data


# Organize all data
p_d2 = organize_by_product(prices_d2)
p_d1 = organize_by_product(prices_d1)
t_d2 = organize_trades(trades_d2)
t_d1 = organize_trades(trades_d1)

products = sorted(set(list(p_d2.keys()) + list(p_d1.keys())))
print(f"\nProducts found: {products}")

# =====================================================================
#  ANALYSIS 1: Basic price statistics
# =====================================================================
# First question: what does each product's price look like?
# - Is it stable or does it move a lot? (std tells us)
# - Does it stay in a narrow range? (min/max tells us)
# - Is the mean the same across days? (if not, price drifts)

print("\n" + "="*60)
print("  ANALYSIS 1: Price Statistics")
print("="*60)

for product in products:
    print(f"\n  {product}:")
    for label, pdata in [("Day -2", p_d2), ("Day -1", p_d1)]:
        if product not in pdata:
            continue
        mids = np.array(pdata[product]['mids']) # mid point between best bid and best ask for that tick
        print(f"    {label}: mean={mids.mean():.1f}  std={mids.std():.2f}  "
              f"range=[{mids.min():.1f}, {mids.max():.1f}]  ticks={len(mids)}")

    # WHAT TO LOOK FOR:
    #   - Low std + same mean across days = fixed fair value (like EMERALDS)
    #   - High std + different means = drifting price, need dynamic fair value

'''
EMERALDS:
    Day -2: mean=10000.0  std=0.73  range=[9996.0, 10004.0]  ticks=10000
    Day -1: mean=10000.0  std=0.72  range=[9996.0, 10004.0]  ticks=10000

TOMATOES:
    Day -2: mean=5007.9  std=10.29  range=[4988.0, 5036.0]  ticks=10000
    Day -1: mean=4977.6  std=14.58  range=[4946.5, 5011.0]  ticks=10000
'''

# =====================================================================
#  ANALYSIS 2: Bid-ask spread
# =====================================================================
# The spread is the gap between the best bid and best ask.
# This matters because:
#   - Your profit per market-making round-trip is at most the spread
#   - You need to place orders INSIDE the spread to get priority
#   - Wider spread = more profit potential but less competition

print("\n" + "="*60)
print("  ANALYSIS 2: Bid-Ask Spread")
print("="*60)

for product in products:
    print(f"\n  {product}:")
    # Combine both days
    all_spreads = []
    for pdata in [p_d2, p_d1]:
        if product in pdata:
            all_spreads.extend([s for s in pdata[product]['spreads'] if s is not None])

    spreads = np.array(all_spreads)
    dist = Counter(all_spreads).most_common(5)
    print(f"    Mean spread: {spreads.mean():.1f}")
    print(f"    Most common values: {dist}")
    print(f"    -> Your resting orders should be placed INSIDE this spread")
    print(f"       e.g., if spread is {dist[0][0]}, quote at ±{dist[0][0]//2 - 1} from fair")


# =====================================================================
#  ANALYSIS 3: Return autocorrelation (MOST IMPORTANT)
# =====================================================================
# "Returns" = price change from one tick to the next.
# "Autocorrelation" = does the current return predict the next one?
#
#   autocorr < 0  --> MEAN REVERTING (price bounces back)
#                     Strategy: market make (buy low, sell high)
#
#   autocorr > 0  --> TRENDING (price keeps going same direction)
#                     Strategy: momentum/trend following
#
#   autocorr ~ 0  --> RANDOM WALK (unpredictable)
#                     Strategy: be careful, less edge available
#
# We check at multiple lags:
#   lag-1: does THIS tick's move predict NEXT tick's move?
#   lag-5: does a move predict what happens 5 ticks later?

print("\n" + "="*60)
print("  ANALYSIS 3: Return Autocorrelation")
print("="*60)

for product in products:
    print(f"\n  {product}:")
    for label, pdata in [("Day -2", p_d2), ("Day -1", p_d1)]:
        if product not in pdata:
            continue
        mids = np.array(pdata[product]['mids'])

        # np.diff computes: [mids[1]-mids[0], mids[2]-mids[1], ...]
        # These are the tick-to-tick price changes ("returns")
        returns = np.diff(mids)

        print(f"    {label}:")
        print(f"      Return std: {returns.std():.3f} (typical tick-to-tick move)")

        # np.corrcoef computes the Pearson correlation between two arrays
        # returns[:-1] = all returns except the last
        # returns[1:]  = all returns except the first
        # Correlating them answers: "when return[t] is positive, is return[t+1] also positive?"
        for lag in [1, 2, 5, 20]:
            if len(returns) > lag:
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                tag = ""
                if lag == 1:
                    if corr < -0.2:
                        tag = " <-- MEAN REVERTING"
                    elif corr > 0.2:
                        tag = " <-- TRENDING"
                    else:
                        tag = " (random walk)"
                print(f"      lag-{lag:>2}: {corr:+.4f}{tag}")

    # WHAT TO LOOK FOR:
    #   Strong negative lag-1 = mean reverting = market making works well
    #   The effect at lag-2+ tells you how fast the reversion happens

'''
============================================================
  ANALYSIS 3: Return Autocorrelation
============================================================

  EMERALDS:
    Day -2:
      Return std: 1.028 (typical tick-to-tick move)
      lag- 1: -0.4848 <-- MEAN REVERTING
      lag- 2: -0.0227
      lag- 5: -0.0015
      lag-20: -0.0091
    Day -1:
      Return std: 0.999 (typical tick-to-tick move)
      lag- 1: -0.4904 <-- MEAN REVERTING
      lag- 2: -0.0048
      lag- 5: -0.0080
      lag-20: -0.0112

  TOMATOES:
    Day -2:
      Return std: 1.345 (typical tick-to-tick move)
      lag- 1: -0.4280 <-- MEAN REVERTING
      lag- 2: -0.0036
      lag- 5: -0.0182
      lag-20: +0.0016
    Day -1:
      Return std: 1.337 (typical tick-to-tick move)
      lag- 1: -0.4125 <-- MEAN REVERTING
      lag- 2: -0.0153
      lag- 5: +0.0112
      lag-20: -0.0256

The only predictable pattern is the one-tick bounce. It shows that mean-version effect exists. 
'''


# =====================================================================
#  ANALYSIS 4: Order book depth
# =====================================================================
# How much volume sits at each price level?
# This tells you:
#   - How much you can aggressively buy/sell per tick (L1 volume)
#   - Whether the book is symmetric (same on bid and ask side)

print("\n" + "="*60)
print("  ANALYSIS 4: Order Book Depth")
print("="*60)

for product in products:
    print(f"\n  {product}:")
    all_bid_vols = []
    all_ask_vols = []
    for pdata in [p_d2, p_d1]:
        if product in pdata:
            all_bid_vols.extend([v for v in pdata[product]['bid1_vols'] if v is not None])
            all_ask_vols.extend([v for v in pdata[product]['ask1_vols'] if v is not None])

    bid_v = np.array(all_bid_vols)
    ask_v = np.array(all_ask_vols)
    print(f"    L1 bid volume: mean={bid_v.mean():.1f}  min={bid_v.min()}  max={bid_v.max()}")
    print(f"    L1 ask volume: mean={ask_v.mean():.1f}  min={ask_v.min()}  max={ask_v.max()}")
    print(f"    Symmetric: {'Yes' if abs(bid_v.mean() - ask_v.mean()) < 1 else 'No'}")
    print(f"    -> Max aggressive fill per tick: ~{int(bid_v.mean())} units")


# # =====================================================================
# #  ANALYSIS 5: Bot trade patterns
# # =====================================================================
# # The bots trade with each other throughout the day. Understanding this:
# #   - Tells you how often resting orders might get filled
# #   - Shows what prices bots actually trade at
# #   - Reveals trade size distribution

# print("\n" + "="*60)
# print("  ANALYSIS 5: Bot Trade Patterns")
# print("="*60)

# for product in products:
#     print(f"\n  {product}:")
#     all_prices = []
#     all_qtys = []
#     all_ts = []
#     for tdata in [t_d2, t_d1]:
#         if product in tdata:
#             all_prices.extend(tdata[product]['prices'])
#             all_qtys.extend(tdata[product]['quantities'])
#             all_ts.extend(tdata[product]['timestamps'])

#     if not all_prices:
#         print("    No trades found")
#         continue

#     prices = np.array(all_prices)
#     qtys = np.array(all_qtys)
#     ts = np.array(all_ts)
#     gaps = np.diff(sorted(ts))

#     print(f"    Total trades: {len(prices)}")
#     print(f"    Total volume: {qtys.sum()}")
#     print(f"    Avg quantity: {qtys.mean():.1f}")
#     print(f"    Qty distribution: {Counter(all_qtys).most_common(5)}")
#     print(f"    Time between trades: mean={gaps.mean():.0f}  median={np.median(gaps):.0f}")

#     # Where do trades actually happen?
#     price_dist = Counter([int(p) for p in all_prices]).most_common(8)
#     print(f"    Most common trade prices: {price_dist}")
#     print(f"    -> Your resting orders get filled when bots trade at YOUR price")


# # =====================================================================
# #  ANALYSIS 6: Intraday price trajectory
# # =====================================================================
# # Does the price tend to go up or down during the day?
# # Split the day into quarters and check the drift in each.

# print("\n" + "="*60)
# print("  ANALYSIS 6: Intraday Trajectory")
# print("="*60)

# for product in products:
#     print(f"\n  {product}:")
#     for label, pdata in [("Day -2", p_d2), ("Day -1", p_d1)]:
#         if product not in pdata:
#             continue
#         mids = np.array(pdata[product]['mids'])
#         n = len(mids)
#         q = n // 4  # quarter size

#         print(f"    {label}:")
#         for i, name in enumerate(["Q1 (start)", "Q2", "Q3", "Q4 (end)"]):
#             segment = mids[i*q : (i+1)*q]
#             drift = segment[-1] - segment[0]
#             print(f"      {name}: {segment[0]:.1f} -> {segment[-1]:.1f}  "
#                   f"(drift={drift:+.1f}, std={segment.std():.2f})")


# # =====================================================================
# #  SUMMARY: Strategy recommendations
# # =====================================================================
# print("\n" + "="*60)
# print("  SUMMARY: Strategy Recommendations")
# print("="*60)

# for product in products:
#     print(f"\n  {product}:")

#     # Check stability
#     all_mids = []
#     for pdata in [p_d2, p_d1]:
#         if product in pdata:
#             all_mids.extend(pdata[product]['mids'])
#     mids = np.array(all_mids)
#     returns = np.diff(mids)
#     ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]

#     all_spreads = []
#     for pdata in [p_d2, p_d1]:
#         if product in pdata:
#             all_spreads.extend([s for s in pdata[product]['spreads'] if s is not None])
#     avg_spread = np.mean(all_spreads)

#     if mids.std() < 2:
#         print(f"    Price type: STABLE (std={mids.std():.2f})")
#         print(f"    -> Use FIXED fair value at {round(mids.mean())}")
#     else:
#         print(f"    Price type: DYNAMIC (std={mids.std():.2f})")
#         print(f"    -> Use EMA to track moving fair value")

#     if ac1 < -0.2:
#         print(f"    Behavior: MEAN REVERTING (autocorr={ac1:.3f})")
#         print(f"    -> Market making strategy (buy low, sell high)")
#     elif ac1 > 0.2:
#         print(f"    Behavior: TRENDING (autocorr={ac1:.3f})")
#         print(f"    -> Momentum strategy (follow the trend)")
#     else:
#         print(f"    Behavior: RANDOM WALK (autocorr={ac1:.3f})")

#     print(f"    Avg spread: {avg_spread:.1f}")
#     suggested_edge = max(1, int(avg_spread // 4))
#     print(f"    -> Suggested MM edge: {suggested_edge} "
#           f"(quote at fair ± {suggested_edge})")

# print()
