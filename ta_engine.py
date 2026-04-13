import numpy as np
import pandas as pd


# Minimum pivot touches required per timeframe to keep a trendline
MIN_TOUCHES = {
    'D': 3,
    'W': 2,
    'M': 2,
}


def find_pivots(df, left=5, right=5):
    """Find swing highs and lows using rolling window."""
    highs = df['High'].values
    lows  = df['Low'].values
    n = len(df)

    pivot_highs = []
    pivot_lows  = []

    for i in range(left, n - right):
        if highs[i] == max(highs[i - left: i + right + 1]):
            pivot_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - left: i + right + 1]):
            pivot_lows.append((i, lows[i]))

    return pivot_highs, pivot_lows


def get_trendlines(pivots, n_lines=2, min_touches=3):
    """
    From a list of (index, price) pivots, return the best-fit trendlines.
    Only returns lines with touches >= min_touches.
    """
    if len(pivots) < 2:
        return []

    lines = []
    for i in range(len(pivots)):
        for j in range(i + 1, len(pivots)):
            x1, y1 = pivots[i]
            x2, y2 = pivots[j]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Count touches (within 0.5% tolerance)
            touches = 0
            for xp, yp in pivots:
                projected = slope * xp + intercept
                if abs(yp - projected) / projected < 0.005:
                    touches += 1

            # Drop weak lines here, before sorting
            if touches < min_touches:
                continue

            lines.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'slope': slope,
                'intercept': intercept,
                'touches': touches
            })

    lines.sort(key=lambda l: l['touches'], reverse=True)

    # Deduplicate — skip lines too similar to an already-selected one
    selected = []
    for l in lines:
        duplicate = False
        for s in selected:
            if abs(l['slope'] - s['slope']) / (abs(s['slope']) + 1e-9) < 0.1:
                duplicate = True
                break
        if not duplicate:
            selected.append(l)
        if len(selected) >= n_lines:
            break

    return selected


def get_sr_levels(df, pivot_highs, pivot_lows, tolerance=0.015):
    """
    Cluster pivot prices into horizontal S/R zones.
    Returns list of price levels sorted descending.
    """
    all_prices = [p for _, p in pivot_highs] + [p for _, p in pivot_lows]
    if not all_prices:
        return []

    all_prices.sort()
    clusters = []
    current_cluster = [all_prices[0]]

    for price in all_prices[1:]:
        if (price - current_cluster[-1]) / current_cluster[-1] < tolerance:
            current_cluster.append(price)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [price]
    clusters.append(np.mean(current_cluster))

    strong = []
    for level in clusters:
        touches = sum(1 for p in all_prices if abs(p - level) / level < tolerance)
        if touches >= 2:
            strong.append(round(level, 2))

    return sorted(strong, reverse=True)


def get_fibonacci(df, lookback=None):
    """
    Auto-detect the biggest swing in the lookback window.
    Returns fib levels dict {ratio: price}.
    """
    if lookback:
        df = df.tail(lookback)

    swing_high = df['High'].max()
    swing_low  = df['Low'].min()
    diff = swing_high - swing_low

    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {}

    high_idx = df['High'].idxmax()
    low_idx  = df['Low'].idxmin()

    if high_idx > low_idx:
        for r in ratios:
            levels[r] = round(swing_high - r * diff, 2)
    else:
        for r in ratios:
            levels[r] = round(swing_low + r * diff, 2)

    return levels, swing_high, swing_low


def detect_channel(resistance_lines, support_lines):
    if not resistance_lines or not support_lines:
        return False, None

    r_slope = resistance_lines[0]['slope']
    s_slope = support_lines[0]['slope']

    if abs(r_slope) < 1e-9:
        return False, None
    if abs((r_slope - s_slope) / r_slope) < 0.20:
        return True, round((r_slope + s_slope) / 2, 6)

    return False, None


def detect_breakout(df, sr_levels, lookback=3):
    if not sr_levels or len(df) < lookback + 1:
        return []

    prev_close = df['Close'].iloc[-(lookback + 1)]
    curr_close = df['Close'].iloc[-1]

    breakouts = []
    for level in sr_levels:
        if prev_close < level <= curr_close:
            breakouts.append({'type': 'BREAKOUT UP', 'level': level})
        elif prev_close > level >= curr_close:
            breakouts.append({'type': 'BREAKOUT DOWN', 'level': level})

    return breakouts


def run_ta(df, timeframe='D'):
    """
    Run full TA pipeline on a OHLCV dataframe.
    timeframe: 'D', 'W', or 'M' — controls min_touches filter.
    Returns dict of all computed signals.
    """
    if len(df) < 20:
        return {}

    mt = MIN_TOUCHES.get(timeframe, 3)

    pivot_highs, pivot_lows = find_pivots(df, left=5, right=5)

    resistance_lines = get_trendlines(pivot_highs, n_lines=2, min_touches=mt)
    support_lines    = get_trendlines(pivot_lows,  n_lines=2, min_touches=mt)

    sr_levels = get_sr_levels(df, pivot_highs, pivot_lows)

    fib_levels, swing_high, swing_low = get_fibonacci(df, lookback=min(252, len(df)))

    is_channel, channel_slope = detect_channel(resistance_lines, support_lines)

    breakouts = detect_breakout(df, sr_levels)

    return {
        'pivot_highs':       pivot_highs,
        'pivot_lows':        pivot_lows,
        'resistance_lines':  resistance_lines,
        'support_lines':     support_lines,
        'sr_levels':         sr_levels,
        'fib_levels':        fib_levels,
        'swing_high':        swing_high,
        'swing_low':         swing_low,
        'is_channel':        is_channel,
        'channel_slope':     channel_slope,
        'breakouts':         breakouts,
    }
