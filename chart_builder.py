import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from ta_engine import run_ta

COLORS = {
    'resistance_line': '#FF4444',
    'support_line':    '#00CC88',
    'sr_level':        '#FFB347',
    'fib':             '#9B59B6',
    'channel':         '#3498DB',
    'pivot_high':      '#FF6B6B',
    'pivot_low':       '#51CF66',
    'breakout_up':     '#00FF88',
    'breakout_down':   '#FF3366',
    'bg':              '#0D1117',
    'grid':            '#21262D',
    'text':            '#C9D1D9',
    'candle_up':       '#26A69A',
    'candle_down':     '#EF5350',
}

FIB_COLORS = {
    0:     '#FFFFFF',
    0.236: '#F39C12',
    0.382: '#E74C3C',
    0.5:   '#9B59B6',
    0.618: '#2ECC71',
    0.786: '#3498DB',
    1.0:   '#FFFFFF',
}


def _add_candlesticks(fig, df, row, col, name=""):
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=name,
        increasing_line_color=COLORS['candle_up'],
        decreasing_line_color=COLORS['candle_down'],
        increasing_fillcolor=COLORS['candle_up'],
        decreasing_fillcolor=COLORS['candle_down'],
        showlegend=False,
        line=dict(width=1),
    ), row=row, col=col)


def _add_trendlines(fig, df, ta, row, col):
    """
    Draw trendlines using ONLY this df's own date index.
    x1 is an integer index into df — clip it to valid range before lookup.
    x2 always extends to df's last date (not beyond).
    No cross-panel date bleed possible.
    """
    dates = df.index
    n = len(dates)

    for i, line in enumerate(ta.get('resistance_lines', [])):
        x1_idx  = min(int(line['x1']), n - 1)   # clip to this df's length
        x1_date = dates[x1_idx]
        x2_date = dates[-1]                      # end = last bar of THIS panel only

        slope     = line['slope']
        intercept = line['intercept']
        y1        = slope * x1_idx + intercept
        y2        = slope * (n - 1) + intercept

        fig.add_trace(go.Scatter(
            x=[x1_date, x2_date],
            y=[y1, y2],
            mode='lines',
            line=dict(color=COLORS['resistance_line'], width=1.5),
            name='Resistance TL',
            showlegend=(row == 1 and col == 1 and i == 0),
        ), row=row, col=col)

    for i, line in enumerate(ta.get('support_lines', [])):
        x1_idx  = min(int(line['x1']), n - 1)
        x1_date = dates[x1_idx]
        x2_date = dates[-1]

        slope     = line['slope']
        intercept = line['intercept']
        y1        = slope * x1_idx + intercept
        y2        = slope * (n - 1) + intercept

        fig.add_trace(go.Scatter(
            x=[x1_date, x2_date],
            y=[y1, y2],
            mode='lines',
            line=dict(color=COLORS['support_line'], width=1.5),
            name='Support TL',
            showlegend=(row == 1 and col == 1 and i == 0),
        ), row=row, col=col)


def _add_sr_levels(fig, df, ta, row, col):
    x_start = df.index[max(0, len(df) - 60)]
    x_end   = df.index[-1]
    current_price = df['Close'].iloc[-1]
    for level in ta.get('sr_levels', []):
        color = COLORS['sr_level']
        fig.add_shape(type='line',
            x0=x_start, x1=x_end, y0=level, y1=level,
            line=dict(color=color, width=1, dash='dot'),
            row=row, col=col)
        label = f"R {level:.0f}" if level >= current_price else f"S {level:.0f}"
        fig.add_annotation(x=x_end, y=level, text=label,
            showarrow=False, font=dict(size=9, color=color),
            xanchor='left', row=row, col=col)


def _add_fibonacci(fig, df, ta, row, col):
    x_start = df.index[0]
    x_end   = df.index[-1]
    for ratio, price in ta.get('fib_levels', {}).items():
        color = FIB_COLORS.get(ratio, '#AAAAAA')
        fig.add_shape(type='line',
            x0=x_start, x1=x_end, y0=price, y1=price,
            line=dict(color=color, width=1, dash='dashdot'),
            row=row, col=col)
        fig.add_annotation(x=x_start, y=price,
            text=f"F{ratio*100:.0f}% {price:.0f}",
            showarrow=False, font=dict(size=8, color=color),
            xanchor='right', row=row, col=col)


def _add_breakout_markers(fig, df, ta, row, col):
    for bo in ta.get('breakouts', []):
        color = COLORS['breakout_up'] if 'UP' in bo['type'] else COLORS['breakout_down']
        fig.add_annotation(x=df.index[-1], y=bo['level'],
            text=f"⚡{bo['type']}",
            showarrow=True, arrowhead=2,
            arrowcolor=color, font=dict(size=10, color=color),
            row=row, col=col)


def _add_pivot_markers(fig, df, ta, row, col):
    dates = df.index
    n = len(dates)
    ph_x = [dates[i] for i, _ in ta.get('pivot_highs', []) if i < n]
    ph_y = [p        for i, p in ta.get('pivot_highs', []) if i < n]
    pl_x = [dates[i] for i, _ in ta.get('pivot_lows',  []) if i < n]
    pl_y = [p        for i, p in ta.get('pivot_lows',  []) if i < n]

    if ph_x:
        fig.add_trace(go.Scatter(x=ph_x, y=ph_y, mode='markers',
            marker=dict(symbol='triangle-down', size=6, color=COLORS['pivot_high']),
            name='Pivot High', showlegend=(row == 1 and col == 1)), row=row, col=col)
    if pl_x:
        fig.add_trace(go.Scatter(x=pl_x, y=pl_y, mode='markers',
            marker=dict(symbol='triangle-up', size=6, color=COLORS['pivot_low']),
            name='Pivot Low', showlegend=(row == 1 and col == 1)), row=row, col=col)


def _build_summary_text(ta, tf_label):
    if not ta:
        return f"<b>{tf_label}</b><br>No data"
    lines = [f"<b>{tf_label}</b>"]
    if ta.get('sr_levels'):
        top3 = ta['sr_levels'][:3]
        lines.append(f"S/R: {', '.join(str(x) for x in top3)}")
    if ta.get('fib_levels'):
        lines.append(f"Fib 61.8%: {ta['fib_levels'].get(0.618, '—')}")
        lines.append(f"Fib 38.2%: {ta['fib_levels'].get(0.382, '—')}")
    if ta.get('resistance_lines'):
        lines.append(f"Res TL slope: {ta['resistance_lines'][0]['slope']:.4f}")
    if ta.get('support_lines'):
        lines.append(f"Sup TL slope: {ta['support_lines'][0]['slope']:.4f}")
    if ta.get('is_channel'):
        lines.append("⬜ Channel detected")
    for b in ta.get('breakouts', []):
        lines.append(f"⚡ {b['type']} @ {b['level']}")
    return "<br>".join(lines)


def build_chart(symbol, df_daily, df_weekly, df_monthly):
    # Pass timeframe to run_ta so min_touches filter is applied correctly
    ta_d = run_ta(df_daily.tail(252),   timeframe='D')
    ta_w = run_ta(df_weekly.tail(104),  timeframe='W')
    ta_m = run_ta(df_monthly.tail(60),  timeframe='M')

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=["Daily (1Y)", "Weekly (2Y)", "Monthly (5Y)"],
        horizontal_spacing=0.06)

    datasets   = [df_daily.tail(252), df_weekly.tail(104), df_monthly.tail(60)]
    ta_results = [ta_d, ta_w, ta_m]
    tf_labels  = ["Daily", "Weekly", "Monthly"]

    for col_idx, (df, ta, tf) in enumerate(zip(datasets, ta_results, tf_labels), start=1):
        if df.empty or not ta:
            continue
        _add_candlesticks(fig, df, row=1, col=col_idx, name=tf)
        _add_trendlines(fig, df, ta, row=1, col=col_idx)
        _add_sr_levels(fig, df, ta, row=1, col=col_idx)
        _add_fibonacci(fig, df, ta, row=1, col=col_idx)
        _add_pivot_markers(fig, df, ta, row=1, col=col_idx)
        _add_breakout_markers(fig, df, ta, row=1, col=col_idx)

    summary_lines = [_build_summary_text(ta, tf) for ta, tf in zip(ta_results, tf_labels)]
    fig.add_annotation(x=0.01, y=0.99, xref='paper', yref='paper',
        text="<br><br>".join(summary_lines),
        showarrow=False, align='left',
        bgcolor='rgba(13,17,23,0.85)', bordercolor='#444', borderwidth=1,
        font=dict(size=9, color=COLORS['text'], family='monospace'),
        xanchor='left', yanchor='top')

    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> — Auto TA | D / W / M",
            font=dict(size=18, color=COLORS['text'], family='monospace')),
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        height=550, width=1600,
        showlegend=True,
        legend=dict(bgcolor='rgba(13,17,23,0.8)', bordercolor='#444',
            borderwidth=1, font=dict(size=9), x=0.01, y=0.02, xanchor='left'),
        margin=dict(l=60, r=40, t=60, b=40),
    )

    axis_style = dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'],
                      showgrid=True, color=COLORS['text'])
    x_extra    = dict(rangeslider=dict(visible=False))

    fig.update_xaxes(**{**axis_style, **x_extra})
    fig.update_yaxes(**axis_style)

    for ann in fig.layout.annotations:
        ann.font.color = COLORS['text']

    return fig
