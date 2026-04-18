from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import json
import math


class Trader:
    POSITION_LIMIT = 80

    OSM = "ASH_COATED_OSMIUM"
    PEP = "INTARIAN_PEPPER_ROOT"

    # Round 2 config — tuned via 2-fold CV on days -1, 0; tested on day 1.
    # Conservative changes from Round 1 baseline (which scored 9,022 real):
    #   - OSMIUM ema_span 15→10 for faster fair value tracking
    #   - PEPPER drift_horizon 70→85 to capture more of the +0.1/tick drift
    #   - PEPPER skew_coef 0.04→0.05 for slightly stronger inventory control
    #   - Taper-in layers: small near fair (adverse selection), large at edges
    CONFIG = {
        OSM: {
            "fair_anchor":        None,
            "anchor_weight":      0.0,
            "drift_per_tick":     0.0,
            "drift_horizon":      0,
            "ema_span":           10,
            "take_edge":          0,
            "skew_coef":          0.04,
            "layers":             [(i, max(4, i)) for i in range(1, 13)],
        },
        PEP: {
            "fair_anchor":        None,
            "anchor_weight":      0.0,
            "drift_per_tick":     0.10,
            "drift_horizon":      85,
            "ema_span":           15,
            "take_edge":          0,
            "skew_coef":          0.05,
            "layers":             [(i, max(4, i+2)) for i in range(1, 11)],
        },
    }

    # MAF: bid for 25% more quote volume.
    # Daily MM profit ~9,000. Extra 25% ≈ 2,250 marginal value.
    # Bid conservatively to land in top 50% while minimizing cost.
    MAF_BID = 1

    def bid(self):
        return self.MAF_BID

    def run(self, state: TradingState):
        saved: dict = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except Exception:
                saved = {}
        saved.setdefault("ema", {})
        saved.setdefault("last_fair", {})

        result: Dict[str, List[Order]] = {}
        for product in state.order_depths:
            if product in self.CONFIG:
                result[product] = self._market_make(state, product, saved)
            else:
                result[product] = []

        return result, 0, json.dumps(saved)

    def _market_make(self, state: TradingState, product: str,
                     saved: dict) -> List[Order]:
        cfg = self.CONFIG[product]
        od = state.order_depths[product]
        position = state.position.get(product, 0)
        start_pos = position

        fair = self._compute_fair(od, product, cfg, saved)
        if fair is None:
            return []

        skewed_fair = fair - position * cfg["skew_coef"]

        buys: List[Order] = []
        sells: List[Order] = []

        take_edge = cfg["take_edge"]
        for ask in sorted(od.sell_orders.keys()):
            if ask <= skewed_fair - take_edge:
                vol = -od.sell_orders[ask]
                qty = min(vol, self.POSITION_LIMIT - position)
                if qty > 0:
                    buys.append(Order(product, ask, qty))
                    position += qty
            else:
                break

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid >= skewed_fair + take_edge:
                vol = od.buy_orders[bid]
                qty = min(vol, self.POSITION_LIMIT + position)
                if qty > 0:
                    sells.append(Order(product, bid, -qty))
                    position -= qty
            else:
                break

        best_bid, best_ask = self._best_bbo(od)
        self._place_layers(
            buys, sells, product, skewed_fair, cfg["layers"],
            best_bid, best_ask, position,
        )

        return self._clamp(buys, sells, start_pos)

    def _compute_fair(self, od: OrderDepth, product: str, cfg: dict,
                      saved: dict) -> Optional[float]:
        best_bid, best_ask = self._best_bbo(od)

        if best_bid is not None and best_ask is not None:
            bv = od.buy_orders[best_bid]
            av = -od.sell_orders[best_ask]
            micro = (best_bid * av + best_ask * bv) / max(bv + av, 1)
        elif best_bid is not None:
            micro = best_bid + 2.0
        elif best_ask is not None:
            micro = best_ask - 2.0
        else:
            return saved["last_fair"].get(product)

        span = cfg["ema_span"]
        alpha = 2.0 / (span + 1)
        prev_ema = saved["ema"].get(product)
        if prev_ema is None:
            ema = micro
        else:
            ema = alpha * micro + (1 - alpha) * prev_ema
        saved["ema"][product] = ema

        if cfg["fair_anchor"] is not None:
            w = cfg["anchor_weight"]
            base_fair = w * cfg["fair_anchor"] + (1 - w) * ema
        else:
            base_fair = ema

        fair = base_fair + cfg["drift_per_tick"] * cfg["drift_horizon"]
        saved["last_fair"][product] = fair
        return fair

    def _place_layers(self, buys: list, sells: list, product: str,
                      fair: float, layers: list,
                      best_bid: Optional[int], best_ask: Optional[int],
                      position: int) -> None:
        prev_bid_px = best_ask if best_ask is not None else int(fair + 4)
        prev_ask_px = best_bid if best_bid is not None else int(fair - 4)

        buy_used = 0
        sell_used = 0
        for edge, size in layers:
            bid_px = min(int(math.floor(fair - edge)), prev_bid_px - 1)
            ask_px = max(int(math.ceil(fair + edge)), prev_ask_px + 1)

            buy_room = self.POSITION_LIMIT - position - buy_used
            sell_room = self.POSITION_LIMIT + position - sell_used

            buy_qty = min(size, max(buy_room, 0))
            sell_qty = min(size, max(sell_room, 0))

            if buy_qty > 0:
                buys.append(Order(product, bid_px, buy_qty))
                buy_used += buy_qty
            if sell_qty > 0:
                sells.append(Order(product, ask_px, -sell_qty))
                sell_used += sell_qty

            prev_bid_px = bid_px
            prev_ask_px = ask_px

    @staticmethod
    def _best_bbo(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def _clamp(self, buys: List[Order], sells: List[Order],
               current_pos: int) -> List[Order]:
        max_buy = self.POSITION_LIMIT - current_pos
        max_sell = self.POSITION_LIMIT + current_pos

        def trim(orders: List[Order], cap: int, sign: int) -> List[Order]:
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

        buys = trim(buys, max_buy, +1)
        sells = trim(sells, max_sell, -1)
        return buys + sells
