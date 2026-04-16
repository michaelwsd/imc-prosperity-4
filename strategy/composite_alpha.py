"""Composite Alpha strategy — fuses five proven quantitative edges into a
single multi-timeframe, volume-aware, microstructure-sensitive signal engine.

Edges combined:
1. Volume-weighted momentum (SSRN #4825389)
2. VPIN-lite order-flow proxy from OHLCV (arXiv 2602.00776)
3. Multi-timeframe confluence
4. Inverse-variance volatility scaling (ScienceDirect July 2025)
5. Improved microstructure (consecutive absorption, momentum decay filter)

Design philosophy:
- PATIENT entries: only trade when multiple independent edges align
- SIMPLE exits: trailing stop does the heavy lifting, let winners ride
- ASYMMETRIC risk/reward: small frequent losses, large infrequent wins
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.core.enums import Side, SignalType
from src.core.models import Candle, Signal
from src.strategy.base import ParamSpec, Strategy, StrategyContext
from src.strategy.indicators import (
    atr,
    bollinger_bands,
    ema,
    rsi,
    sma,
    volume_weighted_momentum,
    vpin_lite,
)


class CompositeAlphaStrategy(Strategy):
    """Multi-edge composite strategy combining institutional microstructure
    reading, volume-weighted momentum, VPIN order-flow toxicity, and
    multi-timeframe confluence with inverse-variance position sizing.

    Entry types (all require multi-TF confluence >= 2):
      1. Absorption -> Impulse: consecutive absorption bars then directional
         impulse with VPIN confirmation.  Full size.
      2. Volume-weighted momentum breakout: VW-return exceeds threshold with
         EMA alignment, volume confirmation, and pressure.  70% size.

    Exits (deliberately simple — let the trailing stop work):
      - Adaptive ATR trailing stop (widens early, tightens in profit)
      - Exhaustion candle (genuine reversal signal with volume)

    Sizing: inverse-variance x confluence multiplier, clamped [0.05, 0.6].
    """

    name = "composite_alpha"
    description = (
        "Multi-edge composite strategy fusing volume-weighted momentum, VPIN "
        "order-flow, multi-timeframe confluence, and improved microstructure "
        "detection with inverse-variance position sizing.  Long & short."
    )

    def __init__(
        self,
        fast_period: int = 8,
        slow_period: int = 21,
        atr_period: int = 14,
        vol_period: int = 20,
        vpin_period: int = 20,
        vwm_period: int = 14,
        bb_period: int = 20,
        mtf_ratio: int = 4,
        htf_ratio: int = 24,
        trail_mult: float = 3.5,
        position_pct: float = 0.20,
        target_variance: float = 0.0004,
    ) -> None:
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            atr_period=atr_period,
            vol_period=vol_period,
            vpin_period=vpin_period,
            vwm_period=vwm_period,
            bb_period=bb_period,
            mtf_ratio=mtf_ratio,
            htf_ratio=htf_ratio,
            trail_mult=trail_mult,
            position_pct=position_pct,
            target_variance=target_variance,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.vol_period = vol_period
        self.vpin_period = vpin_period
        self.vwm_period = vwm_period
        self.bb_period = bb_period
        self.mtf_ratio = mtf_ratio
        self.htf_ratio = htf_ratio
        self.trail_mult = trail_mult
        self.position_pct = position_pct
        self.target_variance = target_variance

    def warmup_periods(self) -> int:
        return self.htf_ratio * max(self.slow_period, self.bb_period, self.vol_period) + 30

    # ── Core candle handler ──────────────────────────────────────────

    def on_candle(self, candle: Candle, ctx: StrategyContext) -> list[Signal]:
        ctx.add_candle(candle)
        if len(ctx.candles) < self.warmup_periods():
            return []

        df = ctx.to_dataframe()
        features = self._compute_features(df, candle)
        if features is None:
            return []

        # Store for ML feature capture
        ctx.metadata["_last_features"] = features

        price = candle.close
        in_position = bool(ctx.metadata.get("in_position", False))

        if in_position:
            ctx.metadata["bars_in_position"] = ctx.metadata.get("bars_in_position", 0) + 1
            return self._check_exits(
                candle, ctx, price, features["atr"],
                features["_upper_wick"], features["_lower_wick"],
                features["_candle_range"],
                features["vol_ratio"], features["bar_direction"],
            )

        return self._check_entries(
            candle, ctx, price, features["atr"], features["body_ratio"],
            features["vol_ratio"], features["pressure"], features["vol_force"],
            features["rsi"], features["_fast_ema"], features["_slow_ema"],
            int(features["consec_abs"]), features["bar_direction"],
            int(features["confluence"]), features["vpin"], features["vwm"],
            features["_vwm_std"], features["iv_scalar"],
        )

    # ── Feature extraction (shared by trading + ML) ──────────────────

    def _compute_features(
        self, df: pd.DataFrame, candle: Candle,
    ) -> dict[str, float] | None:
        """Extract features from current market state.

        Returns a dict of 30 ML features plus internal values (prefixed ``_``)
        needed by entry/exit logic.  Returns ``None`` when data is insufficient
        or contains NaNs in critical indicators.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        open_ = df["open"]

        # ── Indicators ──
        fast_ema_s = ema(close, self.fast_period)
        slow_ema_s = ema(close, self.slow_period)
        atr_s = atr(high, low, close, self.atr_period)
        rsi_s = rsi(close, 14)
        vol_sma_s = sma(volume, self.vol_period)
        vpin_s = vpin_lite(high, low, close, volume, self.vpin_period)
        vwm_s = volume_weighted_momentum(close, volume, self.vwm_period)
        bb = bollinger_bands(close, self.bb_period)

        # ── Multi-timeframe confluence ──
        confluence = self._calc_confluence(df)

        # ── Microstructure ──
        candle_range = high - low
        safe_range = candle_range.replace(0, np.nan)
        body = (close - open_).abs()
        body_ratio = (body / safe_range).fillna(0)

        candle_top = np.maximum(close, open_)
        candle_bot = np.minimum(close, open_)
        upper_wick = high - candle_top
        lower_wick = candle_bot - low

        close_position = ((close - low) / safe_range).fillna(0.5)
        bar_dir = np.sign(close - open_)
        vol_force = bar_dir * body_ratio * volume
        vol_force_smooth = vol_force.rolling(self.fast_period, min_periods=1).mean()

        pressure = (
            (close_position - 0.5) * volume
        ).rolling(self.fast_period, min_periods=1).sum()

        norm_range = (candle_range / atr_s.replace(0, np.nan)).fillna(1)
        vol_ratio_s = (volume / vol_sma_s.replace(0, np.nan)).fillna(1)
        absorption = (vol_ratio_s / norm_range.replace(0, np.nan)).fillna(0)
        consec_abs = self._count_consecutive_absorption(absorption, thresh=1.5)

        # ── Current-bar scalars ──
        price = candle.close
        cur_atr = float(atr_s.iloc[-1])
        cur_body_ratio = float(body_ratio.iloc[-1])
        cur_vol_ratio = float(vol_ratio_s.iloc[-1])
        cur_pressure = float(pressure.iloc[-1])
        cur_vol_force = float(vol_force_smooth.iloc[-1])
        cur_rsi = float(rsi_s.iloc[-1])
        cur_vpin = float(vpin_s.iloc[-1])
        cur_vwm = float(vwm_s.iloc[-1])
        fast_now = float(fast_ema_s.iloc[-1])
        slow_now = float(slow_ema_s.iloc[-1])
        cur_upper_wick = float(upper_wick.iloc[-1])
        cur_lower_wick = float(lower_wick.iloc[-1])
        cur_range = float(candle_range.iloc[-1])
        cur_dir = float(bar_dir.iloc[-1])

        vwm_std = float(vwm_s.rolling(self.vwm_period, min_periods=self.vwm_period).std().iloc[-1]) \
            if len(vwm_s) >= self.vwm_period else 0.01

        iv_scalar = self._inverse_variance_scalar(close)

        # NaN guard
        if any(v != v for v in [cur_atr, cur_rsi, fast_now, slow_now, cur_vpin]):
            return None
        if cur_atr <= 0:
            return None

        # ── Additional ML features ──
        bb_upper = float(bb["upper"].iloc[-1])
        bb_lower = float(bb["lower"].iloc[-1])
        bb_width = bb_upper - bb_lower
        bb_position = (price - bb_lower) / bb_width if bb_width > 0 else 0.5

        ema_spread = (fast_now - slow_now) / cur_atr
        close_ema_dev = (price - fast_now) / cur_atr

        returns_s = close.pct_change().dropna()
        realized_var = float(returns_s.iloc[-20:].var()) if len(returns_s) >= 20 else 0.0
        if realized_var != realized_var:  # NaN guard
            realized_var = 0.0

        vwm_z_score = cur_vwm / max(vwm_std, 1e-8)

        safe_cur_range = cur_range if cur_range > 0 else 1e-8
        upper_wick_ratio = cur_upper_wick / safe_cur_range
        lower_wick_ratio = cur_lower_wick / safe_cur_range

        returns_1h = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else 0.0
        returns_4h = float(close.pct_change(4).iloc[-1]) if len(close) > 4 else 0.0
        returns_24h = float(close.pct_change(24).iloc[-1]) if len(close) > 24 else 0.0
        if returns_1h != returns_1h:
            returns_1h = 0.0
        if returns_4h != returns_4h:
            returns_4h = 0.0
        if returns_24h != returns_24h:
            returns_24h = 0.0

        vol_prev = float(volume.iloc[-2]) if len(volume) > 1 else 0.0
        volume_change = float(volume.iloc[-1]) / vol_prev if vol_prev > 0 else 1.0

        hour = candle.timestamp.hour
        dow = candle.timestamp.weekday()
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin = math.sin(2 * math.pi * dow / 7)
        dow_cos = math.cos(2 * math.pi * dow / 7)

        return {
            # Directional (5)
            "confluence": float(confluence),
            "rsi": cur_rsi,
            "ema_spread": ema_spread,
            "close_ema_dev": close_ema_dev,
            "bar_direction": cur_dir,
            # Volatility (5)
            "atr": cur_atr,
            "realized_var": realized_var,
            "iv_scalar": iv_scalar,
            "norm_range": float(norm_range.iloc[-1]),
            "bb_position": bb_position,
            # Volume/Flow (6)
            "vpin": cur_vpin,
            "vwm": cur_vwm,
            "vwm_z_score": vwm_z_score,
            "vol_ratio": cur_vol_ratio,
            "vol_force": cur_vol_force,
            "pressure": cur_pressure,
            # Microstructure (6)
            "body_ratio": cur_body_ratio,
            "close_position": float(close_position.iloc[-1]),
            "absorption": float(absorption.iloc[-1]),
            "consec_abs": float(consec_abs),
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            # Context (4)
            "returns_1h": returns_1h,
            "returns_4h": returns_4h,
            "returns_24h": returns_24h,
            "volume_change": volume_change,
            # Time (4)
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            # ── Internal (used by strategy routing, not ML) ──
            "_fast_ema": fast_now,
            "_slow_ema": slow_now,
            "_candle_range": cur_range,
            "_upper_wick": cur_upper_wick,
            "_lower_wick": cur_lower_wick,
            "_vwm_std": vwm_std,
        }

    # ── Fill hook ────────────────────────────────────────────────────

    def on_fill(self, fill, ctx: StrategyContext) -> None:  # type: ignore[no-untyped-def]
        current_side = ctx.metadata.get("trade_side")

        if fill.side == Side.BUY:
            if current_side == "short":
                self._clear_state(ctx)
            else:
                self._set_entry_state(ctx, "long", fill.price)
        else:
            if current_side == "long":
                self._clear_state(ctx)
            else:
                self._set_entry_state(ctx, "short", fill.price)

    # ── Multi-timeframe confluence ───────────────────────────────────

    def _calc_confluence(self, df: pd.DataFrame) -> int:
        """Score from -3 to +3 based on trend agreement across 3 timeframes."""
        score = 0
        for ratio in [1, self.mtf_ratio, self.htf_ratio]:
            if ratio == 1:
                tf_df = df
            else:
                tf_df = self._resample(df, ratio)
            if len(tf_df) < self.slow_period + 5:
                continue
            c = tf_df["close"]
            ema_fast = ema(c, min(self.fast_period, len(c) - 1))
            ema_slow = ema(c, min(self.slow_period, len(c) - 1))
            rsi_s = rsi(c, 14)

            cur_fast = float(ema_fast.iloc[-1])
            cur_slow = float(ema_slow.iloc[-1])
            cur_rsi = float(rsi_s.iloc[-1])

            if cur_fast != cur_fast or cur_slow != cur_slow:
                continue

            if cur_fast > cur_slow and cur_rsi > 50:
                score += 1
            elif cur_fast < cur_slow and cur_rsi < 50:
                score -= 1
        return score

    @staticmethod
    def _resample(df: pd.DataFrame, ratio: int) -> pd.DataFrame:
        """Aggregate LTF bars into higher timeframe bars."""
        n = len(df)
        if n < ratio:
            return df
        trim = n % ratio
        trimmed = df.iloc[trim:] if trim else df
        n_groups = len(trimmed) // ratio
        rows = []
        for i in range(n_groups):
            chunk = trimmed.iloc[i * ratio : (i + 1) * ratio]
            rows.append({
                "open": chunk["open"].iloc[0],
                "high": chunk["high"].max(),
                "low": chunk["low"].min(),
                "close": chunk["close"].iloc[-1],
                "volume": chunk["volume"].sum(),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _count_consecutive_absorption(absorption: pd.Series, thresh: float) -> int:
        """Count consecutive absorption bars ending at the second-to-last bar."""
        vals = absorption.values
        count = 0
        for i in range(len(vals) - 2, -1, -1):
            if vals[i] > thresh:
                count += 1
            else:
                break
        return count

    def _inverse_variance_scalar(self, close: pd.Series) -> float:
        """Compute inverse-variance sizing scalar from rolling realized variance."""
        returns = close.pct_change().dropna()
        if len(returns) < 20:
            return 1.0
        realized_var = float(returns.iloc[-20:].var())
        if realized_var <= 0 or realized_var != realized_var:
            return 1.0
        return min(self.target_variance / realized_var, 2.5)

    # ── Entry logic ──────────────────────────────────────────────────

    def _check_entries(
        self,
        candle: Candle,
        ctx: StrategyContext,
        price: float,
        cur_atr: float,
        body_ratio: float,
        vol_ratio: float,
        pressure: float,
        vol_force: float,
        rsi_val: float,
        fast_ema: float,
        slow_ema: float,
        consec_abs: int,
        bar_direction: float,
        confluence: int,
        vpin: float,
        vwm: float,
        vwm_std: float,
        iv_scalar: float,
    ) -> list[Signal]:
        meta: dict[str, Any] = {
            "body_ratio": round(body_ratio, 3),
            "vol_ratio": round(vol_ratio, 2),
            "pressure": round(pressure, 2),
            "confluence": confluence,
            "vpin": round(vpin, 4),
            "vwm": round(vwm, 6),
            "iv_scalar": round(iv_scalar, 3),
            "consec_absorption": consec_abs,
        }

        # ── Confluence gate ──
        if abs(confluence) < 2:
            return []

        is_bullish = confluence >= 2
        is_bearish = confluence <= -2

        # ── TYPE 1: Absorption → Impulse (primary — highest conviction) ──
        is_impulse = body_ratio >= 0.70 and vol_ratio > 1.3
        has_abs = consec_abs >= 2

        if (
            is_bullish and bar_direction > 0
            and is_impulse and has_abs
            and pressure > 0 and vol_force > 0
            and vpin > 0.35
        ):
            meta["entry_type"] = "absorption_impulse_long"
            size = self._calc_size(iv_scalar, confluence)
            self._set_entry_state(ctx, "long", price)
            return [self._signal(candle, SignalType.ENTRY_LONG, Side.BUY, size, meta)]

        if (
            is_bearish and bar_direction < 0
            and is_impulse and has_abs
            and pressure < 0 and vol_force < 0
            and vpin > 0.35
        ):
            meta["entry_type"] = "absorption_impulse_short"
            size = self._calc_size(iv_scalar, confluence)
            self._set_entry_state(ctx, "short", price)
            return [self._signal(candle, SignalType.ENTRY_SHORT, Side.SELL, size, meta)]

        # ── TYPE 2: Volume-Weighted Momentum Breakout ──
        vwm_thresh = 2.0 * max(vwm_std, 1e-8)

        if (
            is_bullish
            and vwm > vwm_thresh
            and price > fast_ema > slow_ema
            and bar_direction > 0
            and pressure > 0
            and vol_ratio > 1.2
            and body_ratio > 0.5
        ):
            meta["entry_type"] = "vwm_breakout_long"
            size = self._calc_size(iv_scalar, confluence) * 0.7
            self._set_entry_state(ctx, "long", price)
            return [self._signal(candle, SignalType.ENTRY_LONG, Side.BUY, size, meta)]

        if (
            is_bearish
            and vwm < -vwm_thresh
            and price < fast_ema < slow_ema
            and bar_direction < 0
            and pressure < 0
            and vol_ratio > 1.2
            and body_ratio > 0.5
        ):
            meta["entry_type"] = "vwm_breakout_short"
            size = self._calc_size(iv_scalar, confluence) * 0.7
            self._set_entry_state(ctx, "short", price)
            return [self._signal(candle, SignalType.ENTRY_SHORT, Side.SELL, size, meta)]

        return []

    # ── Exit logic (deliberately simple) ─────────────────────────────

    def _check_exits(
        self,
        candle: Candle,
        ctx: StrategyContext,
        price: float,
        cur_atr: float,
        upper_wick: float,
        lower_wick: float,
        candle_range: float,
        vol_ratio: float,
        bar_direction: float,
    ) -> list[Signal]:
        trade_side = ctx.metadata.get("trade_side")
        entry_price = ctx.metadata.get("entry_price", price)

        if trade_side == "long":
            best = max(ctx.metadata.get("best_price", price), candle.high)
            ctx.metadata["best_price"] = best
            profit_atr = (best - entry_price) / cur_atr if cur_atr > 0 else 0

            # Adaptive trail: wider early (let trade develop), tightens in profit
            if profit_atr < 2.0:
                trail = self.trail_mult * cur_atr
            else:
                trail = self.trail_mult * 0.7 * cur_atr  # tighten to lock gains
            stop = best - trail

            # Exhaustion: genuine reversal signal only
            wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
            exhaustion = wick_ratio > 0.5 and vol_ratio > 1.5 and bar_direction < 0

            if price <= stop or exhaustion:
                reason = "exhaustion" if exhaustion else "atr_trail"
                meta = {"exit_reason": reason, "stop": round(stop, 4)}
                return [self._signal(candle, SignalType.EXIT_LONG, Side.SELL, 1.0, meta)]

        elif trade_side == "short":
            best = min(ctx.metadata.get("best_price", price), candle.low)
            ctx.metadata["best_price"] = best
            profit_atr = (entry_price - best) / cur_atr if cur_atr > 0 else 0

            if profit_atr < 2.0:
                trail = self.trail_mult * cur_atr
            else:
                trail = self.trail_mult * 0.7 * cur_atr
            stop = best + trail

            wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
            exhaustion = wick_ratio > 0.5 and vol_ratio > 1.5 and bar_direction > 0

            if price >= stop or exhaustion:
                reason = "exhaustion" if exhaustion else "atr_trail"
                meta = {"exit_reason": reason, "stop": round(stop, 4)}
                return [self._signal(candle, SignalType.EXIT_SHORT, Side.BUY, 1.0, meta)]

        return []

    # ── Helpers ──────────────────────────────────────────────────────

    def _signal(
        self, candle: Candle, signal_type: SignalType, side: Side,
        size: float, meta: dict[str, Any],
    ) -> Signal:
        return Signal(
            timestamp=candle.timestamp,
            strategy_id=self.id,
            symbol=candle.symbol,
            signal_type=signal_type,
            side=side,
            suggested_size=size,
            suggested_price=candle.close,
            metadata=meta,
        )

    def _calc_size(self, iv_scalar: float, confluence: int) -> float:
        """Inverse-variance x confluence sizing."""
        conf_mult = {2: 0.7, 3: 1.0}.get(abs(confluence), 0.7)
        size = self.position_pct * iv_scalar * conf_mult
        return float(min(max(size, 0.05), 0.6))

    @staticmethod
    def _set_entry_state(ctx: StrategyContext, side: str, price: float) -> None:
        ctx.metadata["in_position"] = True
        ctx.metadata["trade_side"] = side
        ctx.metadata["entry_price"] = price
        ctx.metadata["best_price"] = price
        ctx.metadata["bars_in_position"] = 0

    @staticmethod
    def _clear_state(ctx: StrategyContext) -> None:
        ctx.metadata["in_position"] = False
        for key in ("trade_side", "entry_price", "best_price", "bars_in_position"):
            ctx.metadata.pop(key, None)

    # ── Parameter spec ───────────────────────────────────────────────

    @classmethod
    def param_spec(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="fast_period", type="int", default=8, min=3, max=20, step=1,
                description="Fast EMA for short-term momentum detection",
            ),
            ParamSpec(
                name="slow_period", type="int", default=21, min=10, max=60, step=1,
                description="Slow EMA trend filter and multi-TF base period",
            ),
            ParamSpec(
                name="atr_period", type="int", default=14, min=5, max=30, step=1,
                description="ATR period for stops and volatility measurement",
            ),
            ParamSpec(
                name="vol_period", type="int", default=20, min=10, max=40, step=1,
                description="Volume moving average period for relative volume",
            ),
            ParamSpec(
                name="vpin_period", type="int", default=20, min=10, max=40, step=1,
                description="VPIN-lite EMA smoothing period",
            ),
            ParamSpec(
                name="vwm_period", type="int", default=14, min=5, max=30, step=1,
                description="Volume-weighted momentum lookback period",
            ),
            ParamSpec(
                name="bb_period", type="int", default=20, min=10, max=40, step=1,
                description="Bollinger Bands period (used in confluence)",
            ),
            ParamSpec(
                name="mtf_ratio", type="int", default=4, min=2, max=12, step=1,
                description="Mid-timeframe aggregation ratio (e.g. 4 = 4xLTF)",
            ),
            ParamSpec(
                name="htf_ratio", type="int", default=24, min=8, max=48, step=1,
                description="High-timeframe aggregation ratio (e.g. 24 = 24xLTF)",
            ),
            ParamSpec(
                name="trail_mult", type="float", default=3.5, min=1.5, max=6.0,
                step=0.1,
                description="ATR multiplier for trailing stop (tightens after profit)",
            ),
            ParamSpec(
                name="position_pct", type="float", default=0.20, min=0.05, max=0.5,
                step=0.05,
                description="Base position size as fraction of equity",
            ),
            ParamSpec(
                name="target_variance", type="float", default=0.0004, min=0.0001,
                max=0.002, step=0.0001,
                description="Target variance for inverse-variance sizing (~2% daily vol squared)",
            ),
        ]
