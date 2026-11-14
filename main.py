"""Mean-reversion bot for the BTC 24h race (agent-fun/bars-replay/v1).

Strategy: buy the dip *after* it turns, not while it is falling.
On every 1-minute bar close the bot tracks, for CBBTC:

  * RSI(14) on closes (Wilder's smoothing), and
  * Bollinger bands on the last 25 closes (mean +/- 1.8 standard deviations).

Entry (long only, at most one open position, 75% of cash): the previous
close sat *below* the lower band while the current close has climbed back
to or above it -- a band re-entry -- and RSI is still under 30, so the
bounce starts from genuinely oversold ground. The order fills at the next
bar's open, so the bot is buying the first minute of the recovery, not the
falling knife itself.

Exit (sell the whole position) on whichever comes first:
  * the close reaches 1.0% above the buy's fill price (take profit),
  * the position's value drops 0.3% below what the buy cost (stop loss),
  * 180 bars have passed since the fill (time stop).

After an exit the bot waits 5 bars before it may enter again, so it cannot
chase the same dip twice. No randomness is used anywhere; every decision is
a pure function of the bars the harness has sent, so the two graded runs
agree bar for bar. Diagnostics go to stderr only -- stdout carries replies.
"""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- tuned parameters (grid-searched on the kit's dev window) -----------------
RSI_BARS = 14          # Wilder RSI lookback, in closes
RSI_OVERSOLD = 30.0    # entry needs RSI below this
BAND_BARS = 25         # Bollinger window, in closes
BAND_STD = 1.8         # band width, in standard deviations
SIZE_NUM = 3           # buy spends SIZE_NUM / SIZE_DEN of cash
SIZE_DEN = 4
TAKE_PROFIT_BPS = 100  # exit when the close is this far above the fill price
STOP_LOSS_BPS = 30     # exit when the position is this far under its cost
MAX_HOLD_BARS = 180    # exit after this many bars held, whatever the price
COOLDOWN_BARS = 5      # bars to wait after an exit before re-entering
MAX_SLIPPAGE_BPS = 150  # bound on the fill price move from the latest close


class MeanReversion:
    """Indicator state and trade state for one symbol, fed bar by bar."""

    def __init__(self):
        self.closes = deque(maxlen=BAND_BARS + 2)
        self._changes = []          # first RSI_BARS closes' changes, pre-seed
        self._avg_gain = None       # Wilder averages, once seeded
        self._avg_loss = None
        self.rsi = None
        self.prev_lower = None      # lower band of the previous bar
        self.lower = None           # lower band of the latest bar
        self.entry_price = None     # fill price of the open position (price units)
        self.entry_cost = 0         # quote units the open position cost
        self.held_bars = 0
        self.cooldown = 0

    def on_close(self, close):
        """Fold one bar close (float, USDC) into RSI and the bands."""
        if self.closes:
            change = close - self.closes[-1]
            if self._avg_gain is None:
                self._changes.append(change)
                if len(self._changes) == RSI_BARS:
                    gains = sum(c for c in self._changes if c > 0.0)
                    losses = sum(-c for c in self._changes if c < 0.0)
                    self._avg_gain = gains / RSI_BARS
                    self._avg_loss = losses / RSI_BARS
                    self.rsi = self._rsi_value()
            else:
                gain = change if change > 0.0 else 0.0
                loss = -change if change < 0.0 else 0.0
                self._avg_gain = (self._avg_gain * (RSI_BARS - 1) + gain) / RSI_BARS
                self._avg_loss = (self._avg_loss * (RSI_BARS - 1) + loss) / RSI_BARS
                self.rsi = self._rsi_value()
        self.closes.append(close)
        if len(self.closes) >= BAND_BARS:
            window = list(self.closes)[-BAND_BARS:]
            mean = sum(window) / BAND_BARS
            var = sum((x - mean) ** 2 for x in window) / BAND_BARS
            self.prev_lower = self.lower
            self.lower = mean - BAND_STD * (var ** 0.5)

    def _rsi_value(self):
        if self._avg_loss == 0.0:
            return 100.0
        if self._avg_gain == 0.0:
            return 0.0
        return 100.0 * self._avg_gain / (self._avg_gain + self._avg_loss)

    @property
    def ready(self):
        return self.rsi is not None and self.lower is not None and self.prev_lower is not None

    def entry_signal(self, prev_close):
        """True on a band re-entry from oversold ground: the previous close
        was under the lower band, this close is back at or above it, and RSI
        is still below the oversold line."""
        return (
            self.ready
            and prev_close is not None
            and prev_close < self.prev_lower
            and self.closes[-1] >= self.lower
            and self.rsi < RSI_OVERSOLD
        )


def on_batch(state, batch):
    bot = state.bot
    orders = []
    # Fills of the previous batch's orders land here: record a buy's fill
    # price and cost so the take-profit and stop-loss have an anchor.
    for fill in batch.get("fills", []):
        if fill.get("symbol") != SYMBOL:
            continue
        if fill.get("status") != "filled":
            continue
        if fill.get("side") == "buy":
            bot.entry_price = int(fill["price"])
            bot.entry_cost = int(fill["notional_quote_units"])
            bot.held_bars = 0
        elif fill.get("side") == "sell":
            bot.entry_price = None
            bot.entry_cost = 0
            bot.held_bars = 0
            bot.cooldown = COOLDOWN_BARS

    if state.warming_up:
        return orders

    bar = state.latest.get(SYMBOL)
    if bar is None:
        return orders
    close_units = int(bar["c"])
    price_scale = state.symbols[SYMBOL]["price_scale"]
    close = close_units / (10 ** price_scale)
    prev_close = bot.closes[-1] if bot.closes else None
    bot.on_close(close)

    holding = state.positions.get(SYMBOL)
    if holding is not None and holding.qty > 0:
        bot.held_bars += 1
        value = state.value(SYMBOL, holding.qty)
        # Anchor the exits on the account's own cost record, not only on the
        # fill message: cost_quote_units is the cash the open position cost.
        # The fill price is still the take-profit anchor when we saw the
        # fill; if it was ever missed, rebuild it from cost over quantity.
        entry_cost = holding.cost
        entry_price = bot.entry_price
        if entry_price is None:
            scales = state.symbols[SYMBOL]
            entry_price = (
                holding.cost
                * 10 ** (scales["qty_scale"] + scales["price_scale"])
                // (holding.qty * 10 ** state.hello["capital"]["quote_scale"])
            )
        take_profit = (
            entry_price is not None
            and close_units * 10_000 >= entry_price * (10_000 + TAKE_PROFIT_BPS)
        )
        stop_loss = value * 10_000 < entry_cost * (10_000 - STOP_LOSS_BPS)
        time_stop = bot.held_bars >= MAX_HOLD_BARS
        if take_profit or stop_loss or time_stop:
            orders.append(
                {
                    "side": "sell",
                    "symbol": SYMBOL,
                    "qty": holding.qty,
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                }
            )
        return orders

    if bot.cooldown > 0:
        bot.cooldown -= 1
        return orders
    if bot.entry_signal(prev_close):
        notional = state.cash * SIZE_NUM // SIZE_DEN
        if notional >= 1_000_000:  # at least 1 USDC, the minimum notional
            orders.append(
                {
                    "side": "buy",
                    "symbol": SYMBOL,
                    "notional_quote_units": notional,
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                }
            )
    return orders


def on_hello(state):
    state.bot = MeanReversion()
    sys.stderr.write(
        "hello: mean-reversion bot ready, seed=%s warmup=%d\n"
        % (state.hello.get("seed"), state.warmup_bars)
    )


makefun_bars.run(on_batch, on_hello=on_hello)
