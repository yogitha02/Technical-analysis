import yfinance as yf
import pandas as pd
import os
import traceback
from datetime import datetime
from chart_builder import build_chart

WATCHLIST = ["INFY", "SBIN", "RELIANCE", "ICICIBANK", "LT"]
OUTPUT_DIR = "charts"

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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*50}")
    print(f"  Auto TA Chart Generator — {today}")
    print(f"  Watchlist: {', '.join(WATCHLIST)}")
    print(f"{'='*50}\n")

    for symbol in WATCHLIST:
        print(f"[{symbol}]")
        df_d, df_w, df_m = fetch_data(symbol)
        if df_d is None or df_d.empty:
            print(f"  Skipping {symbol} — no data\n")
            continue
        print(f"  Running TA...", end=" ")
        try:
            fig = build_chart(symbol, df_d, df_w, df_m)
            print("done")
        except Exception as e:
            print(f"TA FAILED — {e}")
            traceback.print_exc()
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{symbol}_{today}.html")
        fig.write_html(out_path, include_plotlyjs='cdn')
        print(f"  Saved → {out_path}\n")

    print(f"{'='*50}")
    print(f"  Done. Open charts/ folder in your browser.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
