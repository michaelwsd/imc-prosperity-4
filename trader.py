from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json
import math


class Trader:
    POSITION_LIMIT = 80

    # EMERALDS config
    EMERALDS_FAIR = 10000
    EMERALDS_MM_EDGE = 4       # resting orders at 9996/10004

    # TOMATOES config
    TOMATOES_EMA_SPAN = 12     # balanced EMA responsiveness
    TOMATOES_MM_EDGE = 5       # wider resting orders to reduce adverse selection
    TOMATOES_POSITION_SKEW = 0.03  # inventory skew to reduce drawdown

    def run(self, state: TradingState):
        if state.traderData and state.traderData != "":
            saved = json.loads(state.traderData)
        else:
            saved = {"tomatoes_ema": None}

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            if product == "EMERALDS":
                orders = self.trade_emeralds(state, product)
            elif product == "TOMATOES":
                orders = self.trade_tomatoes(state, product, saved)
            else:
                orders = []
            result[product] = orders

        traderData = json.dumps(saved)
        conversions = 0
        return result, conversions, traderData

    # ================================================================
    #  EMERALDS: Fixed fair value market making
    # ================================================================
    def trade_emeralds(self, state: TradingState, product: str) -> List[Order]:
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)

        fair = self.EMERALDS_FAIR
        buy_orders: List[Order] = []
        sell_orders: List[Order] = []

        # ── TAKE: buy anything below fair ──────────────────────────
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price < fair:
                volume = -order_depth.sell_orders[ask_price]
                room = self.POSITION_LIMIT - position
                qty = min(volume, room)
                if qty > 0:
                    buy_orders.append(Order(product, ask_price, qty))
                    position += qty

        # ── TAKE: sell into anything above fair ─────────────────────
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price > fair:
                volume = order_depth.buy_orders[bid_price]
                room = self.POSITION_LIMIT + position
                qty = min(volume, room)
                if qty > 0:
                    sell_orders.append(Order(product, bid_price, -qty))
                    position -= qty

        # ── MAKE: place resting orders inside the bot spread ────────
        buy_price = fair - self.EMERALDS_MM_EDGE       # 9996
        sell_price = fair + self.EMERALDS_MM_EDGE       # 10004

        buy_room = self.POSITION_LIMIT - position
        sell_room = self.POSITION_LIMIT + position

        if buy_room > 0:
            buy_orders.append(Order(product, buy_price, buy_room))
        if sell_room > 0:
            sell_orders.append(Order(product, sell_price, -sell_room))

        return self.clamp_orders(buy_orders, sell_orders,
                                 state.position.get(product, 0))

    # ================================================================
    #  TOMATOES: EMA-based mean reversion with inventory skew
    # ================================================================
    def trade_tomatoes(self, state: TradingState, product: str, saved: dict) -> List[Order]:
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        # ── Calculate fair value via EMA of microprice ──────────────
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = -order_depth.sell_orders[best_ask]
        # Microprice: weighted toward the side with more volume
        microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

        alpha = 2 / (self.TOMATOES_EMA_SPAN + 1)
        if saved["tomatoes_ema"] is None:
            saved["tomatoes_ema"] = microprice
        else:
            saved["tomatoes_ema"] = alpha * microprice + (1 - alpha) * saved["tomatoes_ema"]

        # Skew fair value against inventory to encourage position reduction
        fair = saved["tomatoes_ema"] - position * self.TOMATOES_POSITION_SKEW

        buy_orders: List[Order] = []
        sell_orders: List[Order] = []

        # ── TAKE: buy anything below fair ───────────────────────────
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price < fair:
                volume = -order_depth.sell_orders[ask_price]
                room = self.POSITION_LIMIT - position
                qty = min(volume, room)
                if qty > 0:
                    buy_orders.append(Order(product, ask_price, qty))
                    position += qty

        # ── TAKE: sell above fair ───────────────────────────────────
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price > fair:
                volume = order_depth.buy_orders[bid_price]
                room = self.POSITION_LIMIT + position
                qty = min(volume, room)
                if qty > 0:
                    sell_orders.append(Order(product, bid_price, -qty))
                    position -= qty

        # ── MAKE: resting orders around skewed fair ─────────────────
        buy_price = math.floor(fair - self.TOMATOES_MM_EDGE)
        sell_price = math.ceil(fair + self.TOMATOES_MM_EDGE)

        buy_room = self.POSITION_LIMIT - position
        sell_room = self.POSITION_LIMIT + position

        if buy_room > 0:
            buy_orders.append(Order(product, buy_price, buy_room))
        if sell_room > 0:
            sell_orders.append(Order(product, sell_price, -sell_room))

        return self.clamp_orders(buy_orders, sell_orders,
                                 state.position.get(product, 0))

    # ================================================================
    #  Clamp orders so buys and sells each independently respect limits
    # ================================================================
    def clamp_orders(self, buy_orders: List[Order], sell_orders: List[Order],
                     current_pos: int) -> List[Order]:
        """
        The exchange checks: could ALL buys fill? If so, would position
        exceed the limit? Same check for sells independently.

        We cap total buy qty to (LIMIT - position) and total sell qty
        to (LIMIT + position). If over, we trim the LAST order (the
        resting one) since aggressive orders are higher priority.
        """
        max_buy = self.POSITION_LIMIT - current_pos
        max_sell = self.POSITION_LIMIT + current_pos

        # Clamp buys
        total_buy = sum(o.quantity for o in buy_orders)
        if total_buy > max_buy:
            excess = total_buy - max_buy
            # Trim from last order first (resting order)
            for i in range(len(buy_orders) - 1, -1, -1):
                if excess <= 0:
                    break
                reduce = min(excess, buy_orders[i].quantity)
                new_qty = buy_orders[i].quantity - reduce
                if new_qty > 0:
                    buy_orders[i] = Order(buy_orders[i].symbol, buy_orders[i].price, new_qty)
                else:
                    buy_orders[i] = None
                excess -= reduce
            buy_orders = [o for o in buy_orders if o is not None]

        # Clamp sells
        total_sell = sum(-o.quantity for o in sell_orders)
        if total_sell > max_sell:
            excess = total_sell - max_sell
            for i in range(len(sell_orders) - 1, -1, -1):
                if excess <= 0:
                    break
                reduce = min(excess, -sell_orders[i].quantity)
                new_qty = -sell_orders[i].quantity - reduce
                if new_qty > 0:
                    sell_orders[i] = Order(sell_orders[i].symbol, sell_orders[i].price, -new_qty)
                else:
                    sell_orders[i] = None
                excess -= reduce
            sell_orders = [o for o in sell_orders if o is not None]

        return buy_orders + sell_orders
