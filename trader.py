from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import math


class Trader:
    """Round 1 trader for ASH_COATED_OSMIUM and INTARIAN_PEPPER_ROOT.

    Empirical findings (from 3 days of round1 data):

      OSMIUM
        - Fair value stable around 10000 (std ~5)
        - Modal spread 16; bot flow concentrates at ±8 from mid
        - Return autocorrelation ≈ -0.5 (tick-by-tick reversion)
        - Solution: fixed fair + TAKE edges + LAYERED resting quotes
          to intercept flow inside the bot spread.

      PEPPER
        - Deterministic linear drift of +0.1 / tick (≈+1000/day)
        - Noise std ≈ 3/tick around drift, autocorr ≈ -0.5
        - Bot flow clusters near mid (|dev|avg ~4-5)
        - Solution: microprice EMA + drift forecast + LONG-BIASED
          inventory target to capture the drift while market-making.

    Position limit: 80 per product. Products with one-sided books
    (~8% of ticks) fall back to last-known fair value via traderData.
    """

    POSITION_LIMIT = 80

    # ── OSMIUM config ──────────────────────────────────────────────
    OSM = "ASH_COATED_OSMIUM"
    OSM_FAIR = 10000
    OSM_TAKE_EDGE = 1          # take asks <= fair-1, bids >= fair+1
    OSM_SKEW = 0.04            # shift fair down by skew*position
    # Four layers — maximise flow capture across widths
    OSM_LAYERS = [(1, 10), (2, 15), (3, 20), (5, 35)]  # (edge, size) per side

    # ── PEPPER config ──────────────────────────────────────────────
    PEP = "INTARIAN_PEPPER_ROOT"
    PEP_EMA_SPAN = 8           # fast EMA — mid is rising +0.1/tick
    PEP_DRIFT_PER_TICK = 0.10  # empirically measured, +1000/day
    PEP_DRIFT_HORIZON = 30     # anticipate 30 ticks of drift into fair
    PEP_INVENTORY_TARGET = 70  # lean heavily long to harvest drift
    PEP_SKEW = 0.03            # tiny skew — let inventory ride the drift
    PEP_BUY_TAKE_EDGE = 0      # take any ask at or below drift-adjusted fair
    PEP_SELL_TAKE_EDGE = 3     # only sell if bid is 3 above fair (drift will catch up)
    PEP_LAYERS = [(1, 15), (3, 25), (5, 40)]  # (edge, size) per side

    def run(self, state: TradingState):
        saved = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except Exception:
                saved = {}
        saved.setdefault("pep_ema", None)
        saved.setdefault("osm_fair", float(self.OSM_FAIR))

        result: Dict[str, List[Order]] = {}
        for product in state.order_depths:
            if product == self.OSM:
                result[product] = self.trade_osmium(state, product, saved)
            elif product == self.PEP:
                result[product] = self.trade_pepper(state, product, saved)
            else:
                result[product] = []

        return result, 0, json.dumps(saved)

    # ================================================================
    #  OSMIUM — layered market making around a stable fair value
    # ================================================================
    def trade_osmium(self, state: TradingState, product: str, saved: dict) -> List[Order]:
        od = state.order_depths[product]
        position = state.position.get(product, 0)
        start_pos = position

        # Build fair: base 10000, gently updated if mid persistently drifts.
        best_bid, best_ask = self._best_bbo(od)
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
            # Slow EMA update — 0.5% weight — so fair barely moves unless
            # persistent asymmetry is present.
            saved["osm_fair"] = 0.995 * saved["osm_fair"] + 0.005 * mid

        fair_mid = saved["osm_fair"]
        # Inventory skew: pushes quotes away from inventory side
        fair = fair_mid - position * self.OSM_SKEW

        buys, sells = [], []

        # ── TAKE: any ask strictly better than (fair - edge) ─────────
        for ask in sorted(od.sell_orders.keys()):
            if ask <= fair - self.OSM_TAKE_EDGE:
                vol = -od.sell_orders[ask]
                qty = min(vol, self.POSITION_LIMIT - position)
                if qty > 0:
                    buys.append(Order(product, ask, qty))
                    position += qty
            else:
                break

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid >= fair + self.OSM_TAKE_EDGE:
                vol = od.buy_orders[bid]
                qty = min(vol, self.POSITION_LIMIT + position)
                if qty > 0:
                    sells.append(Order(product, bid, -qty))
                    position -= qty
            else:
                break

        # ── MAKE: multi-layer quoting ───────────────────────────────
        book_best_bid = best_bid if best_bid is not None else int(fair_mid - 4)
        book_best_ask = best_ask if best_ask is not None else int(fair_mid + 4)

        self._layered_quote(
            buys, sells, product, fair, self.OSM_LAYERS,
            book_best_bid, book_best_ask, position,
        )

        return self.clamp_orders(buys, sells, start_pos)

    # ================================================================
    #  PEPPER — drift-aware long-biased market making
    # ================================================================
    def trade_pepper(self, state: TradingState, product: str, saved: dict) -> List[Order]:
        od = state.order_depths[product]
        position = state.position.get(product, 0)
        start_pos = position

        best_bid, best_ask = self._best_bbo(od)
        if best_bid is None and best_ask is None:
            return []

        # Compute microprice (volume-weighted mid). Fall back when one-sided.
        if best_bid is not None and best_ask is not None:
            bv = od.buy_orders[best_bid]
            av = -od.sell_orders[best_ask]
            micro = (best_bid * av + best_ask * bv) / max(bv + av, 1)
        elif best_bid is not None:
            micro = best_bid + 2  # assume fair ~2 above lonely bid
        else:
            micro = best_ask - 2

        # EMA of microprice
        alpha = 2 / (self.PEP_EMA_SPAN + 1)
        if saved["pep_ema"] is None:
            saved["pep_ema"] = micro
        else:
            saved["pep_ema"] = alpha * micro + (1 - alpha) * saved["pep_ema"]

        # Drift-adjusted fair: forecast N ticks ahead. This is the expected
        # mid price over the holding horizon; it's our reservation price.
        fair_mid = saved["pep_ema"] + self.PEP_DRIFT_PER_TICK * self.PEP_DRIFT_HORIZON

        # Inventory skew relative to LONG target. If we're below target, we
        # "want" to buy → fair shifts up (more willing to pay).
        deviation = position - self.PEP_INVENTORY_TARGET
        fair = fair_mid - deviation * self.PEP_SKEW

        buys, sells = [], []

        # ── TAKE asks: asymmetric — aggressive buying while drift is up ──
        for ask in sorted(od.sell_orders.keys()):
            if ask <= fair - self.PEP_BUY_TAKE_EDGE:
                vol = -od.sell_orders[ask]
                qty = min(vol, self.POSITION_LIMIT - position)
                if qty > 0:
                    buys.append(Order(product, ask, qty))
                    position += qty
            else:
                break

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid >= fair + self.PEP_SELL_TAKE_EDGE:
                vol = od.buy_orders[bid]
                qty = min(vol, self.POSITION_LIMIT + position)
                if qty > 0:
                    sells.append(Order(product, bid, -qty))
                    position -= qty
            else:
                break

        # ── MAKE: multi-layer around drift-adjusted fair ─────────────
        book_best_bid = best_bid if best_bid is not None else int(fair_mid - 4)
        book_best_ask = best_ask if best_ask is not None else int(fair_mid + 4)

        self._layered_quote(
            buys, sells, product, fair, self.PEP_LAYERS,
            book_best_bid, book_best_ask, position,
        )

        return self.clamp_orders(buys, sells, start_pos)

    # ================================================================
    #  Helpers
    # ================================================================
    @staticmethod
    def _best_bbo(od: OrderDepth):
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _place_layer(dest: list, product: str, price: int, desired: int,
                     room: int, side: int) -> None:
        qty = min(desired, max(room, 0))
        if qty <= 0:
            return
        dest.append(Order(product, price, qty * side))

    def _layered_quote(self, buys, sells, product, fair, layers,
                       book_best_bid, book_best_ask, position):
        """Post a set of (edge, size) layers on each side, respecting book."""
        prev_bid = book_best_ask  # seed so first bid stays ≤ best_ask - 1
        prev_ask = book_best_bid
        buy_used = 0
        sell_used = 0
        for edge, size in layers:
            bid_px = min(int(math.floor(fair - edge)), prev_bid - 1)
            ask_px = max(int(math.ceil(fair + edge)), prev_ask + 1)
            buy_room = self.POSITION_LIMIT - position - buy_used
            sell_room = self.POSITION_LIMIT + position - sell_used
            self._place_layer(buys, product, bid_px, size, buy_room, side=+1)
            self._place_layer(sells, product, ask_px, size, sell_room, side=-1)
            # Re-measure how much we've committed so far
            buy_used = sum(o.quantity for o in buys if o.quantity > 0)
            sell_used = sum(-o.quantity for o in sells if o.quantity < 0)
            prev_bid = bid_px
            prev_ask = ask_px

    def clamp_orders(self, buy_orders: List[Order], sell_orders: List[Order],
                     current_pos: int) -> List[Order]:
        """Respect the exchange's per-side hypothetical-fill position check.

        Total buy qty cannot exceed LIMIT - position; total sell qty cannot
        exceed LIMIT + position. If violated, trim from the last-appended
        orders (resting ones) since aggressive orders are higher priority.
        """
        max_buy = self.POSITION_LIMIT - current_pos
        max_sell = self.POSITION_LIMIT + current_pos

        def trim(orders, cap, sign):
            total = sum(sign * o.quantity for o in orders)
            if total <= cap:
                return orders
            excess = total - cap
            for i in range(len(orders) - 1, -1, -1):
                if excess <= 0:
                    break
                cur = sign * orders[i].quantity
                red = min(excess, cur)
                new_qty = (cur - red) * sign
                if new_qty != 0:
                    orders[i] = Order(orders[i].symbol, orders[i].price, new_qty)
                else:
                    orders[i] = None
                excess -= red
            return [o for o in orders if o is not None]

        buy_orders = trim(buy_orders, max_buy, +1)
        sell_orders = trim(sell_orders, max_sell, -1)
        return buy_orders + sell_orders
