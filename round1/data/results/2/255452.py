from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import json
import math


class Trader:
    """Market making strategy for Round 1.

    Pure two-sided market making, not directional. Profit comes from
    capturing the bid-ask spread on many small trades while keeping
    inventory near zero via skew.

    Two products, one unified engine with per-product configuration:

      ASH_COATED_OSMIUM
        - Stable fair around 10000 (slow-drift EMA anchor)
        - Strong mean reversion (ac1 ~ -0.5)
        - Modal spread 16, occasional widening to 18-21
        - No drift — truly stationary

      INTARIAN_PEPPER_ROOT
        - Deterministic +0.1/tick drift (empirically verified across
          all 3 days, every 10% segment — no intraday variation)
        - Strong mean reversion (ac1 ~ -0.5) on top of the drift
        - Modal spread 11-13, occasional widening to 14-17
        - Fair = EMA(microprice) projected forward by drift*horizon

    Position limit: 80 per product. Target inventory = 0.
    Skew pushes quotes against inventory to naturally unwind.
    """

    POSITION_LIMIT = 80

    OSM = "ASH_COATED_OSMIUM"
    PEP = "INTARIAN_PEPPER_ROOT"

    # ── Per-product MM configuration ────────────────────────────────
    # - fair_anchor: reference fair (None = pure EMA, no anchor)
    # - anchor_weight: blend weight for anchor vs EMA (higher = more sticky)
    # - drift_per_tick: known deterministic drift rate
    # - drift_horizon: ticks ahead to forecast (fair = EMA + drift*horizon)
    # - ema_span: smoothing for microprice EMA
    # - take_edge: take asks <= fair - take_edge, bids >= fair + take_edge
    # - skew_coef: inventory skew (fair shifts by skew * position)
    # - layers: [(edge, size), ...] resting orders per side
    # - max_book_half_spread: cap widest quote width regardless of book
    # Research notes from real-exchange analysis (day 0 actual = 9,022):
    #   - PEPPER captures 97% of pure drift (7.77/tick vs 8.0 theoretical).
    #     Little room for further improvement on drift capture.
    #   - OSMIUM was the bottleneck — only 14% of total profit at 1.27/tick.
    #   - True OSMIUM fair ≈ 10001 (anchored mid-mode), not 10000 — the
    #     hardcoded 10000 anchor was fighting EMA's correct discovery.
    #
    # Optimized choices (validated via 2-fold CV on days -2, -1, and
    # held-out test on day 0):
    #   1. OSMIUM: no anchor, pure EMA fair-value discovery (+ ~6,000 vs anchored)
    #   2. OSMIUM: dense 12-layer quoting — capture many small fills
    #      across the full bot spread
    #   3. PEPPER: dense 10-layer quoting for same reason
    #   4. Both: aggressive take_edge=0 (mean reversion is strong, lag1 ~ -0.5)
    CONFIG = {
        OSM: {
            "fair_anchor":        None,    # pure EMA — data-driven fair
            "anchor_weight":      0.0,
            "drift_per_tick":     0.0,
            "drift_horizon":      0,
            "ema_span":           15,
            "take_edge":          0,
            "skew_coef":          0.04,
            "layers":             [(i, 8) for i in range(1, 13)],  # 12 dense layers
        },
        PEP: {
            "fair_anchor":        None,
            "anchor_weight":      0.0,
            "drift_per_tick":     0.10,
            "drift_horizon":      70,
            "ema_span":           8,
            "take_edge":          0,
            "skew_coef":          0.04,
            "layers":             [(i, 10) for i in range(1, 11)],  # 10 dense layers
        },
    }

    # ================================================================
    # Entry point
    # ================================================================
    def run(self, state: TradingState):
        saved: dict = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except Exception:
                saved = {}
        saved.setdefault("ema", {})        # per-product EMA state
        saved.setdefault("last_fair", {})  # fallback fair when book missing

        result: Dict[str, List[Order]] = {}
        for product in state.order_depths:
            if product in self.CONFIG:
                result[product] = self._market_make(state, product, saved)
            else:
                result[product] = []

        return result, 0, json.dumps(saved)

    # ================================================================
    # Unified market-making engine
    # ================================================================
    def _market_make(self, state: TradingState, product: str,
                     saved: dict) -> List[Order]:
        cfg = self.CONFIG[product]
        od = state.order_depths[product]
        position = state.position.get(product, 0)
        start_pos = position

        fair = self._compute_fair(od, product, cfg, saved)
        if fair is None:
            return []  # no book, no trades

        # Apply inventory skew (push quotes away from our position)
        skewed_fair = fair - position * cfg["skew_coef"]

        buys: List[Order] = []
        sells: List[Order] = []

        # ── TAKE: aggressive fills when book is clearly mispriced ───
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

        # ── MAKE: layered resting quotes ────────────────────────────
        best_bid, best_ask = self._best_bbo(od)
        self._place_layers(
            buys, sells, product, skewed_fair, cfg["layers"],
            best_bid, best_ask, position,
        )

        return self._clamp(buys, sells, start_pos)

    # ================================================================
    # Fair value: hybrid (microprice + wall_mid) EMA + drift forecast
    #
    # Wall mid idea from Frankfurt Hedgehogs (Prosperity 3, 2nd place):
    # averaging the deepest visible bid and ask gives a more stable
    # estimate than L1 microprice alone. L1 is noisy from bot
    # overbidding; deep levels reflect real market-maker quotes.
    # We average 50/50 for the best of both: stability + responsiveness.
    # ================================================================
    def _compute_fair(self, od: OrderDepth, product: str, cfg: dict,
                      saved: dict) -> Optional[float]:
        bids = list(od.buy_orders.keys())
        asks = list(od.sell_orders.keys())

        if bids and asks:
            best_bid, best_ask = max(bids), min(asks)
            bv = od.buy_orders[best_bid]
            av = -od.sell_orders[best_ask]
            micro = (best_bid * av + best_ask * bv) / max(bv + av, 1)
            wall_mid = (min(bids) + max(asks)) / 2
            price_input = 0.5 * micro + 0.5 * wall_mid
        elif bids:
            price_input = max(bids) + 2.0
        elif asks:
            price_input = min(asks) - 2.0
        else:
            return saved["last_fair"].get(product)

        # Update EMA
        ema_key = product
        span = cfg["ema_span"]
        alpha = 2.0 / (span + 1)
        prev_ema = saved["ema"].get(ema_key)
        if prev_ema is None:
            ema = price_input
        else:
            ema = alpha * price_input + (1 - alpha) * prev_ema
        saved["ema"][ema_key] = ema

        if cfg["fair_anchor"] is not None:
            w = cfg["anchor_weight"]
            base_fair = w * cfg["fair_anchor"] + (1 - w) * ema
        else:
            base_fair = ema

        fair = base_fair + cfg["drift_per_tick"] * cfg["drift_horizon"]
        saved["last_fair"][product] = fair
        return fair

    # ================================================================
    # Layered quoting with position-aware asymmetric sizing
    #
    # When long, we shrink bid-side sizes (to slow further accumulation)
    # and keep ask-side full (to unwind fast). Mirror for short.
    # Values tuned via CV: strength 0.8, floor 0.2.
    # ================================================================
    ASYM_STRENGTH = 0.8   # how strongly position skews layer sizes
    ASYM_FLOOR = 0.2      # minimum scale factor — never fully zero a side

    def _place_layers(self, buys: list, sells: list, product: str,
                      fair: float, layers: list,
                      best_bid: Optional[int], best_ask: Optional[int],
                      position: int) -> None:
        pos_ratio = position / self.POSITION_LIMIT  # -1 to 1
        buy_scale = max(self.ASYM_FLOOR,
                        1 - max(pos_ratio, 0) * self.ASYM_STRENGTH)
        sell_scale = max(self.ASYM_FLOOR,
                         1 + min(pos_ratio, 0) * self.ASYM_STRENGTH)

        prev_bid_px = best_ask if best_ask is not None else int(fair + 4)
        prev_ask_px = best_bid if best_bid is not None else int(fair - 4)

        buy_used = 0
        sell_used = 0
        for edge, size in layers:
            bid_px = min(int(math.floor(fair - edge)), prev_bid_px - 1)
            ask_px = max(int(math.ceil(fair + edge)), prev_ask_px + 1)

            buy_room = self.POSITION_LIMIT - position - buy_used
            sell_room = self.POSITION_LIMIT + position - sell_used

            buy_qty = min(int(size * buy_scale), max(buy_room, 0))
            sell_qty = min(int(size * sell_scale), max(sell_room, 0))

            if buy_qty > 0:
                buys.append(Order(product, bid_px, buy_qty))
                buy_used += buy_qty
            if sell_qty > 0:
                sells.append(Order(product, ask_px, -sell_qty))
                sell_used += sell_qty

            prev_bid_px = bid_px
            prev_ask_px = ask_px

    # ================================================================
    # Helpers
    # ================================================================
    @staticmethod
    def _best_bbo(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def _clamp(self, buys: List[Order], sells: List[Order],
               current_pos: int) -> List[Order]:
        """Ensure total buy/sell qty respects position limits even if
        every order fills. Trim the last (widest, most speculative)
        layers first when over-limit.
        """
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
                    orders[i] = None  # type: ignore
                excess -= red
            return [o for o in orders if o is not None]

        buys = trim(buys, max_buy, +1)
        sells = trim(sells, max_sell, -1)
        return buys + sells