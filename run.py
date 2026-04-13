import yfinance as yf
import pandas as pd
import os
import traceback
import base64
import json
import time
import requests
from datetime import datetime
from chart_builder import build_chart
from ta_engine import run_ta

WATCHLIST = ["INFY", "SBIN", "RELIANCE", "ICICIBANK", "LT"]
OUTPUT_DIR = "charts"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY
)

GEMINI_PROMPT = """You are an expert technical analyst. Analyze this {tf} timeframe candlestick chart for an Indian listed stock.

Give exactly 3 bullet points, each one line, no fluff:
• S/R: [is price near a key support or resistance level? which one and how close]
• Structure: [trendlines, channels, Fibonacci levels — what is price doing relative to them]
• Pattern: [any chart pattern visible — H&S, inverse H&S, double top/bottom, wedge, triangle, flag, pennant, cup & handle, or none]

Be specific with price levels where visible. If nothing notable, say so plainly."""


def fetch_data(symbol):
    ticker_sym = f"{symbol}.NS"
    print(f"  Fetching {ticker_sym}...", end=" ")
    try:
        ticker = yf.Ticker(ticker_sym)
        df_d = ticker.history(period="2y",  interval="1d")
        df_w = ticker.history(period="5y",  interval="1wk")
        df_m = ticker.history(period="10y", interval="1mo")
        for df in [df_d, df_w, df_m]:
            df.index = pd.to_datetime(df.index).tz_convert(None)
        print(f"D:{len(df_d)} W:{len(df_w)} M:{len(df_m)} bars")
        return df_d, df_w, df_m
    except Exception as e:
        print(f"FAILED — {e}")
        traceback.print_exc()
        return None, None, None


def save_png(fig, path):
    """Save plotly figure as PNG using kaleido."""
    try:
        fig.write_image(path, width=1600, height=550, scale=1)
        return True
    except Exception as e:
        print(f"  PNG save failed — {e}")
        return False


def build_single_tf_chart(symbol, df, tf_label, timeframe):
    """Build a single-panel chart for one timeframe."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from chart_builder import (COLORS, FIB_COLORS,
                                _add_candlesticks, _add_trendlines,
                                _add_sr_levels, _add_fibonacci,
                                _add_pivot_markers, _add_breakout_markers)

    ta = run_ta(df, timeframe=timeframe)
    if not ta:
        return None

    fig = make_subplots(rows=1, cols=1)
    _add_candlesticks(fig, df, row=1, col=1, name=tf_label)
    _add_trendlines(fig, df, ta, row=1, col=1)
    _add_sr_levels(fig, df, ta, row=1, col=1)
    _add_fibonacci(fig, df, ta, row=1, col=1)
    _add_pivot_markers(fig, df, ta, row=1, col=1)
    _add_breakout_markers(fig, df, ta, row=1, col=1)

    fig.update_layout(
        title=dict(text=f"{symbol} — {tf_label}",
                   font=dict(size=14, color=COLORS['text'], family='monospace')),
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        height=550, width=1600,
        showlegend=False,
        margin=dict(l=60, r=40, t=50, b=40),
    )
    fig.update_xaxes(
        gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'],
        showgrid=True, color=COLORS['text'],
        rangeslider=dict(visible=False)
    )
    fig.update_yaxes(
        gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'],
        showgrid=True, color=COLORS['text']
    )
    return fig


def gemini_analyze(png_path, tf_label):
    """Send PNG to Gemini Flash vision, get 3-line TA summary."""
    if not GEMINI_API_KEY:
        return "Gemini API key not set"
    if not os.path.exists(png_path):
        return "Chart image not found"

    with open(png_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": GEMINI_PROMPT.format(tf=tf_label)},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}}
            ]
        }]
    }

    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
    except Exception as e:
        return f"Gemini error: {e}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*50}")
    print(f"  Auto TA Chart Generator — {today}")
    print(f"  Watchlist: {', '.join(WATCHLIST)}")
    print(f"{'='*50}\n")

    summaries = {}  # symbol -> {D: text, W: text, M: text}

    for symbol in WATCHLIST:
        print(f"[{symbol}]")
        df_d, df_w, df_m = fetch_data(symbol)
        if df_d is None or df_d.empty:
            print(f"  Skipping {symbol} — no data\n")
            continue

        # ── Build combined 3-panel HTML chart ─────────────────────────────
        print(f"  Building HTML chart...", end=" ")
        try:
            fig = build_chart(symbol, df_d, df_w, df_m)
            out_html = os.path.join(OUTPUT_DIR, f"{symbol}_{today}.html")
            fig.write_html(out_html, include_plotlyjs='cdn')
            print("done")
        except Exception as e:
            print(f"FAILED — {e}")
            traceback.print_exc()
            continue

        # ── Build individual timeframe PNGs + Gemini analysis ─────────────
        tf_configs = [
            (df_d.tail(252),  "Daily",   "D"),
            (df_w.tail(104),  "Weekly",  "W"),
            (df_m.tail(60),   "Monthly", "M"),
        ]

        symbol_summary = {}
        for df_tf, tf_label, tf_code in tf_configs:
            print(f"  Gemini {tf_label}...", end=" ")
            png_path = os.path.join(OUTPUT_DIR, f"{symbol}_{tf_code}_tmp.png")

            # Save single-tf PNG
            tf_fig = build_single_tf_chart(symbol, df_tf, tf_label, tf_code)
            if tf_fig is None:
                symbol_summary[tf_code] = "Insufficient data"
                print("skipped")
                continue

            saved = save_png(tf_fig, png_path)
            if not saved:
                symbol_summary[tf_code] = "PNG generation failed"
                print("png failed")
                continue

            analysis = gemini_analyze(png_path, tf_label)
            symbol_summary[tf_code] = analysis
            print("done")

            # Clean up tmp PNG
            try:
                os.remove(png_path)
            except Exception:
                pass

            time.sleep(5)  # avoid rate limit

        summaries[symbol] = symbol_summary
        print()

    # ── Save summaries JSON (for index page) ──────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "summaries.json")
    with open(summary_path, "w") as f:
        json.dump({"date": today, "summaries": summaries}, f, indent=2)

    # ── Build index.html ──────────────────────────────────────────────────
    index_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(index_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
<title>Auto TA — {today}</title>
<style>
  body {{ background:#0D1117; color:#C9D1D9; font-family:monospace; padding:24px; }}
  h2 {{ color:#58A6FF; }}
  .stock {{ border:1px solid #21262D; border-radius:6px; padding:16px; margin-bottom:16px; }}
  .stock h3 {{ margin:0 0 8px 0; }}
  .stock h3 a {{ color:#58A6FF; text-decoration:none; font-size:16px; }}
  .stock h3 a:hover {{ text-decoration:underline; }}
  .tf-block {{ margin:6px 0; }}
  .tf-label {{ color:#FFB347; font-weight:bold; font-size:11px; }}
  .tf-text {{ color:#C9D1D9; font-size:12px; white-space:pre-wrap; margin-left:8px; }}
  .date {{ color:#666; font-size:12px; margin-bottom:20px; }}
</style>
</head>
<body>
<h2>Auto TA Charts</h2>
<div class="date">Generated: {today}</div>
""")

        for symbol in WATCHLIST:
            chart_file = f"{symbol}_{today}.html"
            chart_exists = os.path.exists(os.path.join(OUTPUT_DIR, chart_file))
            sym_summary = summaries.get(symbol, {})

            f.write(f'<div class="stock">\n')
            if chart_exists:
                f.write(f'<h3><a href="{chart_file}">{symbol} ↗</a></h3>\n')
            else:
                f.write(f'<h3>{symbol} — no data</h3>\n')

            for tf_code, tf_label in [("D", "DAILY"), ("W", "WEEKLY"), ("M", "MONTHLY")]:
                text = sym_summary.get(tf_code, "—")
                f.write(f'<div class="tf-block">')
                f.write(f'<span class="tf-label">{tf_label}</span>')
                f.write(f'<span class="tf-text">{text}</span>')
                f.write(f'</div>\n')

            f.write('</div>\n')

        f.write("</body></html>")

    print(f"{'='*50}")
    print(f"  Index → {index_path}")
    print(f"  Done.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
