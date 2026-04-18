"""
Analysis script for IMC Prosperity 4 Round 1.

Usage:
    cd round1 && python analysis.py
"""

import csv
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter


def load_prices(filepath):
    rows = []
    with open(filepath) as f:
        for row in csv.DictReader(f, delimiter=';'):
            rows.append(row)
    return rows


def load_trades(filepath):
    rows = []
    with open(filepath) as f:
        for row in csv.DictReader(f, delimiter=';'):
            rows.append(row)
    return rows


def organize_by_product(price_rows):
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


prices_d2 = load_prices("data/prices_round_0_day_-2.csv")
prices_d1 = load_prices("data/prices_round_0_day_-1.csv")
trades_d2 = load_trades("data/trades_round_0_day_-2.csv")
trades_d1 = load_trades("data/trades_round_0_day_-1.csv")

print(f"Loaded {len(prices_d2)} price rows (day -2), {len(prices_d1)} (day -1)")
print(f"Loaded {len(trades_d2)} trade rows (day -2), {len(trades_d1)} (day -1)")

p_d2 = organize_by_product(prices_d2)
p_d1 = organize_by_product(prices_d1)
t_d2 = organize_trades(trades_d2)
t_d1 = organize_trades(trades_d1)

products = sorted(set(list(p_d2.keys()) + list(p_d1.keys())))
print(f"\nProducts found: {products}")

print("\n" + "="*60)
print("  ANALYSIS 1: Price Statistics")
print("="*60)
for product in products:
    print(f"\n  {product}:")
    for label, pdata in [("Day -2", p_d2), ("Day -1", p_d1)]:
        if product not in pdata:
            continue
        mids = np.array(pdata[product]['mids'])
        print(f"    {label}: mean={mids.mean():.1f}  std={mids.std():.2f}  "
              f"range=[{mids.min():.1f}, {mids.max():.1f}]  ticks={len(mids)}")

print("\n" + "="*60)
print("  ANALYSIS 2: Bid-Ask Spread")
print("="*60)
for product in products:
    print(f"\n  {product}:")
    all_spreads = []
    for pdata in [p_d2, p_d1]:
        if product in pdata:
            all_spreads.extend([s for s in pdata[product]['spreads'] if s is not None])
    spreads = np.array(all_spreads)
    dist = Counter(all_spreads).most_common(5)
    print(f"    Mean spread: {spreads.mean():.1f}")
    print(f"    Most common values: {dist}")

print("\n" + "="*60)
print("  ANALYSIS 3: Return Autocorrelation")
print("="*60)
for product in products:
    print(f"\n  {product}:")
    for label, pdata in [("Day -2", p_d2), ("Day -1", p_d1)]:
        if product not in pdata:
            continue
        mids = np.array(pdata[product]['mids'])
        returns = np.diff(mids)
        print(f"    {label}:")
        print(f"      Return std: {returns.std():.3f}")
        for lag in [1, 2, 5, 20]:
            if len(returns) > lag:
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                tag = ""
                if lag == 1:
                    if corr < -0.2:
                        tag = " <-- MEAN REVERTING"
                    elif corr > 0.2:
                        tag = " <-- TRENDING"
                print(f"      lag-{lag:>2}: {corr:+.4f}{tag}")

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
