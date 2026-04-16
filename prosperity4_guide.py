"""
╔══════════════════════════════════════════════════════════════════════╗
║          IMC PROSPERITY 4 — COMPLETE LEARNING GUIDE                ║
║          How the Datamodels & Trader Class Work                    ║
╚══════════════════════════════════════════════════════════════════════╝

Run this file: python prosperity4_guide.py
It will print a full walkthrough with real examples from the tutorial data.
No actual trading happens — this is purely for learning.
"""

from datamodel import (
    Listing, OrderDepth, Trade, TradingState, Order,
    Observation, ConversionObservation
)
import json

# ════════════════════════════════════════════════════════════════════
# SECTION 1: THE ORDER — Your basic building block
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 1: THE ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

An Order is how you tell the exchange "I want to buy/sell something."
It has exactly 3 fields:

    Order(symbol, price, quantity)

    symbol   → which product (e.g. "EMERALDS")
    price    → the price you're willing to trade at
    quantity → positive = BUY, negative = SELL
""")

# Create some example orders
buy_order = Order("EMERALDS", 9995, 10)     # Buy 10 EMERALDS at 9995
sell_order = Order("EMERALDS", 10005, -10)  # Sell 10 EMERALDS at 10005

print(f"  Buy order:  {buy_order}")
print(f"  Sell order: {sell_order}")
print(f"""
  How to read these:
  • {buy_order} → "Buy 10 EMERALDS, willing to pay up to 9995 each"
  • {sell_order} → "Sell 10 EMERALDS, won't accept less than 10005 each"

  THE KEY RULE: quantity > 0 means BUY, quantity < 0 means SELL.
  That's it. There is no separate "side" field — the sign IS the side.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 2: THE ORDER BOOK (OrderDepth) — What's available to trade
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 2: THE ORDER BOOK (OrderDepth)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The OrderDepth shows you ALL outstanding orders from bots for a product.
Think of it as looking at the screen at a stock exchange.

It has two dicts:
    buy_orders:  {price: quantity}   ← bots wanting to BUY  (positive qty)
    sell_orders: {price: quantity}   ← bots wanting to SELL (NEGATIVE qty!)

⚠️  GOTCHA: sell_orders quantities are NEGATIVE. This trips people up.
""")

# Let's build the EMERALDS order book from actual tutorial data (day -1, timestamp 0)
emeralds_book = OrderDepth()
emeralds_book.buy_orders = {9992: 14, 9990: 29}       # Bots want to buy
emeralds_book.sell_orders = {10008: -14, 10010: -29}   # Bots want to sell

print("  EMERALDS order book (from tutorial day -1, timestamp 0):")
print(f"  buy_orders:  {emeralds_book.buy_orders}")
print(f"  sell_orders: {emeralds_book.sell_orders}")
print("""
  Visualized as a real order book:

       SELL SIDE (asks — bots offering to sell to you)
       ┌──────────┬──────────┐
       │  Price   │ Quantity │
       ├──────────┼──────────┤
       │  10010   │    29    │  ← worse price (further from mid)
       │  10008   │    14    │  ← best ask (cheapest you can buy at)
       └──────────┴──────────┘
       - - - - spread = 16 - - - -
       ┌──────────┬──────────┐
       │   9992   │    14    │  ← best bid (highest someone will pay)
       │   9990   │    29    │  ← worse price (further from mid)
       └──────────┴──────────┘
       BUY SIDE (bids — bots wanting to buy from you)

  The "spread" is the gap between best bid and best ask: 10008 - 9992 = 16
  The "midprice" is the average: (10008 + 9992) / 2 = 10000
""")

# Show how to extract useful info from the book
best_bid = max(emeralds_book.buy_orders.keys())
best_ask = min(emeralds_book.sell_orders.keys())
midprice = (best_bid + best_ask) / 2

print(f"  Extracting key values:")
print(f"    best_bid  = max(buy_orders.keys())  = {best_bid}")
print(f"    best_ask  = min(sell_orders.keys())  = {best_ask}")
print(f"    midprice  = (best_bid + best_ask) / 2 = {midprice}")
print(f"    spread    = best_ask - best_bid       = {best_ask - best_bid}")
print(f"""
  To get the VOLUME available at the best ask:
    raw = sell_orders[{best_ask}] = {emeralds_book.sell_orders[best_ask]}  (negative!)
    actual quantity = abs({emeralds_book.sell_orders[best_ask]}) = {abs(emeralds_book.sell_orders[best_ask])}

  You MUST negate sell quantities to get the actual amount available.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 3: THE TRADE — A completed transaction
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 3: THE TRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A Trade represents something that already happened — a completed transaction.

    Trade(symbol, price, quantity, buyer, seller, timestamp)

You see these in two places:
  • state.own_trades[product]    → trades YOUR algorithm did
  • state.market_trades[product] → trades between OTHER participants (bots)
""")

# Your algo bought 5 EMERALDS at 9992
my_trade = Trade("EMERALDS", 9992, 5, buyer="SUBMISSION", seller="", timestamp=1000)
print(f"  Your trade:   {my_trade}")
print(f"""
  "SUBMISSION" means YOU. That's how the system labels your algo.

  • buyer="SUBMISSION"  → you bought
  • seller="SUBMISSION" → you sold
  • buyer="" or seller="" → counterparty identity is hidden

  In later rounds, bot names may be revealed (e.g. "Olivia", "Pablo").
  Until then, other participants show as empty strings.
""")

# A market trade between bots
bot_trade = Trade("EMERALDS", 9992, 8, buyer="", seller="", timestamp=3200)
print(f"  Bot trade:    {bot_trade}")
print("""
  This is from the actual tutorial data — at timestamp 3200,
  8 EMERALDS traded at 9992 between bots. You can use market_trades
  to understand price dynamics and bot behavior.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 4: LISTINGS — Product metadata
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 4: LISTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A Listing tells you metadata about each tradeable product.

    Listing(symbol, product, denomination)

    symbol       → the ticker name used in orders ("EMERALDS")
    product      → the product name (usually same as symbol)
    denomination → what currency it's priced in ("XIRECS")
""")

listing = Listing(symbol="EMERALDS", product="EMERALDS", denomination="XIRECS")
print(f"  Example: symbol={listing.symbol}, product={listing.product}, denomination={listing.denomination}")
print("""
  In Prosperity 4, everything is denominated in XIRECS.
  You mostly won't need to interact with Listings directly —
  the important stuff is in order_depths, position, and trades.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 5: OBSERVATIONS — External data signals
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 5: OBSERVATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Observations provide external data relevant to certain products.
Two types:

  1. plainValueObservations: simple {product: number} dict
     e.g. {"STARFRUIT_INDEX": 5042}

  2. conversionObservations: complex data for cross-market trading
     Contains bid/ask prices, transport fees, tariffs, sunlight, humidity
""")

# Example conversion observation
conv = ConversionObservation(
    bidPrice=9990.0, askPrice=10010.0,
    transportFees=5.0, exportTariff=2.0, importTariff=3.0,
    sunlight=72.5, humidity=45.0
)
print(f"  Example ConversionObservation:")
print(f"    bidPrice={conv.bidPrice}, askPrice={conv.askPrice}")
print(f"    transportFees={conv.transportFees}")
print(f"    exportTariff={conv.exportTariff}, importTariff={conv.importTariff}")
print(f"    sunlight={conv.sunlight}, humidity={conv.humidity}")
print("""
  Conversions let you trade positions across markets.
  You return an integer "conversions" from run() to request one.
  Rules:
    • You must already hold a position to convert
    • Can't convert more than you hold
    • You pay transport fees + tariffs
    • Return 0 if you don't want to convert

  Not all rounds use conversions. The tutorial round doesn't.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 6: TRADING STATE — Everything combined, your input each tick
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 6: TRADING STATE — The complete picture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every time run() is called, you receive a TradingState.
It bundles EVERYTHING you need to make a decision:
""")

# Build a realistic TradingState from tutorial data
tomatoes_book = OrderDepth()
tomatoes_book.buy_orders = {4999: 5, 4998: 15}
tomatoes_book.sell_orders = {5013: -5, 5014: -15}

state = TradingState(
    traderData="",                                  # empty on first tick
    timestamp=100,
    listings={
        "EMERALDS": Listing("EMERALDS", "EMERALDS", "XIRECS"),
        "TOMATOES": Listing("TOMATOES", "TOMATOES", "XIRECS"),
    },
    order_depths={
        "EMERALDS": emeralds_book,
        "TOMATOES": tomatoes_book,
    },
    own_trades={
        "EMERALDS": [],
        "TOMATOES": [],
    },
    market_trades={
        "EMERALDS": [],
        "TOMATOES": [],
    },
    position={
        "EMERALDS": 0,
        "TOMATOES": 0,
    },
    observations=Observation({}, {}),
)

print(f"""  state.timestamp = {state.timestamp}
    → Current simulation time. Increments by 100 each tick.
      Tick 0, 100, 200, ... up to 99900 (1000 ticks in testing).

  state.traderData = "{state.traderData}"
    → Your saved state from last tick (empty string on first tick).

  state.position = {state.position}
    → How much of each product you currently hold.
      Positive = long (you own it), Negative = short (you owe it).

  state.order_depths = {{product: OrderDepth, ...}}
    → The order book for each product (see Section 2).

  state.own_trades = {{product: [Trade, ...], ...}}
    → What YOUR algo traded since last tick.
      Empty on first tick or if you didn't trade.

  state.market_trades = {{product: [Trade, ...], ...}}
    → What BOTS traded with each other since last tick.
      Useful for reading market activity.

  state.listings = {{symbol: Listing, ...}}
    → Metadata (see Section 4). Rarely needed directly.

  state.observations = Observation(...)
    → External signals (see Section 5). Empty in tutorial round.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 7: THE TRADER CLASS — How it all fits together
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 7: THE TRADER CLASS — Putting it all together
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your Trader class is the ONLY thing you upload. It must:
  1. Be called "Trader" (exact name)
  2. Have a run() method that takes TradingState and returns 3 things
  3. Have a bid() method (only matters in Round 2)

Here's the skeleton:

┌─────────────────────────────────────────────────────────────────┐
│  from datamodel import OrderDepth, TradingState, Order         │
│  from typing import List                                       │
│                                                                │
│  class Trader:                                                 │
│      def bid(self):                                            │
│          return 0                                              │
│                                                                │
│      def run(self, state: TradingState):                       │
│          result = {}          # your orders                    │
│          conversions = 0      # conversion requests            │
│          traderData = ""      # state to save for next tick    │
│                                                                │
│          # ... your logic here ...                             │
│                                                                │
│          return result, conversions, traderData                │
└─────────────────────────────────────────────────────────────────┘

The three return values:

  result: Dict[str, List[Order]]
    → A dictionary mapping product names to lists of orders.
      e.g. {{"EMERALDS": [Order("EMERALDS", 9995, 5)],
             "TOMATOES": [Order("TOMATOES", 5010, -3)]}}

  conversions: int
    → How many units to convert (0 if not using conversions).

  traderData: str
    → Any data you want to persist to the next tick.
      Encode with jsonpickle.encode() or json.dumps().
      Max 50,000 characters before it gets truncated.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 8: POSITION LIMITS — The most common mistake
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 8: POSITION LIMITS — The #1 mistake people make
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each product has a max absolute position. E.g. if limit = 50:
  • Your position must stay between -50 and +50

THE CRITICAL RULE:
  If your total buy (or sell) orders COULD push you past the limit
  — assuming ALL orders fill — then ALL your orders get CANCELLED.
  Not just the excess. ALL of them. For that product, that tick.
""")

# Walk through a concrete example
pos_limit = 50
current_pos = 35

max_buy = pos_limit - current_pos    # 50 - 35 = 15
max_sell = pos_limit + current_pos   # 50 + 35 = 85

print(f"  Example: position limit = {pos_limit}, current position = {current_pos}")
print(f"")
print(f"    Max you can BUY:  {pos_limit} - {current_pos} = {max_buy}")
print(f"    Max you can SELL: {pos_limit} + {current_pos} = {max_sell}")
print(f"")

# Show what happens with bad orders
print(f"    If you send: [Order('EMERALDS', 9990, 10), Order('EMERALDS', 9992, 8)]")
print(f"    Total buy qty: 10 + 8 = 18")
print(f"    Current pos + 18 = {current_pos} + 18 = {current_pos + 18} > {pos_limit}")
print(f"    ❌ ALL ORDERS CANCELLED! You trade nothing this tick.")
print(f"")
print(f"    If you send: [Order('EMERALDS', 9990, 10), Order('EMERALDS', 9992, 5)]")
print(f"    Total buy qty: 10 + 5 = 15")
print(f"    Current pos + 15 = {current_pos} + 15 = {current_pos + 15} = {pos_limit}")
print(f"    ✅ Legal! (exactly at the limit is fine)")
print()

# Show the safe way to calculate
print("""  SAFE PATTERN — always calculate room before ordering:

    position = state.position.get(product, 0)
    buy_room = POSITION_LIMIT - position       # max you can buy
    sell_room = POSITION_LIMIT + position       # max you can sell (abs short)

    # Then cap every order:
    buy_qty = min(desired_qty, buy_room)
    sell_qty = min(desired_qty, sell_room)
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 9: ORDER MATCHING — What happens to your orders
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 9: ORDER MATCHING — What happens after you send orders
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When you return orders from run(), here's what the exchange does:

  IMMEDIATE MATCHING:
  Your buy at price P matches any bot sell at price ≤ P.
  Your sell at price P matches any bot buy at price ≥ P.
  You trade at THE BOT'S price (not yours), which is better for you.

  LEFTOVER QUANTITY:
  If your order is bigger than what's available, the excess becomes
  a "resting order" visible to bots. They MAY trade against it.

  END OF TICK:
  Any remaining resting quantity is CANCELLED. Nothing carries over.
""")

print("""  WALKTHROUGH with EMERALDS order book:

    Sell orders: {10008: -14, 10010: -29}
    Buy orders:  {9992: 14, 9990: 29}

    You send: Order("EMERALDS", 10010, 20)  ← Buy 20 at up to 10010

    Step 1: Match with best ask first
            Buy 14 at 10008 (bot's price — cheaper than your limit!)
            Remaining: 20 - 14 = 6

    Step 2: Match with next ask
            Buy 6 at 10010
            Remaining: 0

    Result: You bought 14 @ 10008 and 6 @ 10010
            Average price: (14×10008 + 6×10010) / 20 = 10008.6

    Next tick, state.own_trades["EMERALDS"] will show these two trades.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 10: traderData — Persisting state between ticks
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 10: traderData — Your memory across ticks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The simulation is STATELESS. Class variables, globals — nothing persists.
traderData is your ONLY way to remember things.

The cycle:
  Tick 1: state.traderData = ""  (empty on first call)
          You return traderData = '{"prices": [10000]}'

  Tick 2: state.traderData = '{"prices": [10000]}'
          You decode it, append new price, return updated string

  Tick 3: state.traderData = '{"prices": [10000, 10002]}'
          ...and so on
""")

import jsonpickle

# Demonstrate the encode/decode cycle
my_state = {
    "price_history": [10000, 10002, 9998, 10001],
    "ema": 10000.5,
    "tick_count": 4
}

encoded = jsonpickle.encode(my_state)
decoded = jsonpickle.decode(encoded)

print(f"  Original:  {my_state}")
print(f"  Encoded:   {encoded}")
print(f"  Decoded:   {decoded}")
print(f"  Same data? {my_state == decoded}")
print(f"  String length: {len(encoded)} characters (limit: 50,000)")
print()
print("""  TIPS:
  • For simple dicts/lists, json.dumps/loads works fine and is faster
  • jsonpickle handles complex objects (custom classes, numpy arrays)
  • ALWAYS cap your stored data (e.g. keep last 100 prices only)
  • If the string exceeds 50,000 chars, it gets TRUNCATED silently
    and jsonpickle.decode() will crash on the broken string
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 11: FULL SIMULATION WALKTHROUGH
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 11: FULL SIMULATION WALKTHROUGH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Let's trace through 3 ticks with EMERALDS to see everything connect.
Position limit: 50. Starting position: 0.
""")

print("""  ═══ TICK 1 (timestamp=0) ═══

  state.position = {"EMERALDS": 0}
  state.traderData = ""
  state.order_depths["EMERALDS"]:
      buy_orders:  {9992: 14, 9990: 29}
      sell_orders: {10008: -14, 10010: -29}
  state.own_trades = {"EMERALDS": []}

  Your logic: "Midprice is 10000. Sell side at 10008 is too expensive.
               I'll place a resting buy at 9995 for 10 units, hoping
               a bot sells to me cheap."

  You return:
      result = {"EMERALDS": [Order("EMERALDS", 9995, 10)]}
      conversions = 0
      traderData = '{"ema": 10000}'

  What happens: Your buy at 9995 doesn't match any sell order
  (cheapest sell is 10008). It becomes a resting quote.
  Bots see it and... one bot sells 6 to you at 9995!

  ═══ TICK 2 (timestamp=100) ═══

  state.position = {"EMERALDS": 6}    ← was 0, bought 6
  state.traderData = '{"ema": 10000}'
  state.own_trades["EMERALDS"] = [
      Trade("EMERALDS", 9995, 6, buyer="SUBMISSION", seller="")
  ]
  state.order_depths["EMERALDS"]:
      buy_orders:  {9993: 12, 9990: 22}      ← book has changed
      sell_orders: {10008: -11, 10010: -22}

  Your logic: "I bought at 9995, now I want to sell above 10000
               for profit. I'll hit the best bid? No, 9993 is below
               my buy price. I'll place a resting sell at 10005."

  Buy room:  50 - 6 = 44
  Sell room: 50 + 6 = 56

  You return:
      result = {"EMERALDS": [Order("EMERALDS", 10005, -6)]}
      conversions = 0
      traderData = '{"ema": 10000.2}'

  What happens: Resting sell at 10005. A bot buys 6 from you!

  ═══ TICK 3 (timestamp=200) ═══

  state.position = {"EMERALDS": 0}    ← sold all 6 back
  state.own_trades["EMERALDS"] = [
      Trade("EMERALDS", 10005, 6, buyer="", seller="SUBMISSION")
  ]

  Profit: Bought 6 @ 9995, sold 6 @ 10005 = 6 × 10 = 60 XIRECS!
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 12: TUTORIAL ROUND PRODUCTS — What you're trading
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 12: TUTORIAL ROUND PRODUCTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

From the sample data (data/tutorial/), the tutorial has 2 products:

  EMERALDS
    • Midprice: exactly 10000, never moves
    • Spread: always 16 (bids at 9990-9992, asks at 10008-10010)
    • Very stable — classic market making target
    • Similar to "Rainforest Resin" from Prosperity 3

  TOMATOES
    • Midprice: starts around 5000, drifts up and down
    • Spread: varies (roughly 14-16)
    • More volatile — price actually moves between ticks
    • Similar to "Kelp" from Prosperity 3

  Currency: XIRECS (replaces SeaShells from previous years)
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 13: CHEAT SHEET
# ════════════════════════════════════════════════════════════════════

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CHEAT SHEET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GET BEST BID:      max(order_depth.buy_orders.keys())
  GET BEST ASK:      min(order_depth.sell_orders.keys())
  GET BID VOLUME:    order_depth.buy_orders[price]        (positive)
  GET ASK VOLUME:    abs(order_depth.sell_orders[price])   (negate it!)
  GET MIDPRICE:      (best_bid + best_ask) / 2
  GET POSITION:      state.position.get(product, 0)
  BUY ORDER:         Order(product, price, +quantity)
  SELL ORDER:        Order(product, price, -quantity)
  MAX BUY QTY:       POSITION_LIMIT - current_position
  MAX SELL QTY:      POSITION_LIMIT + current_position
  SAVE STATE:        traderData = jsonpickle.encode(my_dict)
  LOAD STATE:        my_dict = jsonpickle.decode(state.traderData)
  DEBUG:             print() statements appear in the log file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
