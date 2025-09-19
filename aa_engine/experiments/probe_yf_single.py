# experiments/probe_yf_single.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Please: pip install yfinance")

EXPECTED = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    # If yfinance returns 2-level columns: (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Expect exactly one ticker in the 2nd level
        lvl1 = df.columns.get_level_values(1)
        uniq = list(pd.unique(lvl1))
        if len(uniq) != 1:
            raise ValueError(f"Unexpected multi-ticker frame from yfinance: {uniq}")
        # Keep the FIELD level as columns
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Ensure all expected fields exist
    for c in EXPECTED:
        if c not in df.columns:
            df[c] = pd.NA

    # Normalize index
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    # Rename & order to our package schema
    df = df.rename(columns={"Adj Close": "AdjClose"})
    cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    print(f"[probe] downloading {args.ticker} …")
    raw = yf.download(
        args.ticker, start=args.start, end=args.end, auto_adjust=False, progress=False
    )

    print("\n--- RAW ---")
    print("index:", type(raw.index), "tz:", getattr(raw.index.tz, "zone", None))
    print("columns:", list(raw.columns))
    print("dtypes:\n", raw.dtypes)
    print(raw.head(3))

    can = canonicalize(raw.copy())
    print("\n--- CANONICAL ---")
    print("columns:", list(can.columns))
    print("dtypes:\n", can.dtypes)
    print(can.head(3))

    # Ensure output dir exists
    Path("data/probe").mkdir(parents=True, exist_ok=True)
    out = f"data/probe/{args.ticker.replace(':','_').replace('/','_')}.csv"
    can.reset_index().to_csv(out, index=False)
    print(f"\n[probe] wrote canonical CSV → {out}")


if __name__ == "__main__":
    main()
