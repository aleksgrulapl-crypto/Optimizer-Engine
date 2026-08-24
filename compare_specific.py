# compare_specific.py
import pandas as pd
from pathlib import Path

TV_PATH = Path("AutoTrader_15M_NASDAQ_NVDA_2026-08-20.csv")
OUR_PATH = Path("nvda_test_tvparams.csv")   # file you just created

def load_tv(path):
    encs = ["utf-8-sig","utf-8","latin1","cp1252"]
    for e in encs:
        try:
            df = pd.read_csv(path, dtype=str, encoding=e)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError("Cannot read TV CSV")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "trade_number" in df.columns:
        df["trade_number"] = pd.to_numeric(df["trade_number"], errors="coerce").astype("Int64")
    # parse date with dayfirst
    time_cols = [c for c in df.columns if "time" in c or "date" in c]
    if time_cols:
        df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], dayfirst=True, errors="coerce")
    # canonical price/pnl
    for col in df.columns:
        if "price" in col and "price" not in df.columns:
            df["price"] = pd.to_numeric(df[col].str.replace(",",""), errors="coerce")
    if "price" not in df.columns:
        # try common names
        for k in ("price_usd","price_usd."):
            if k in df.columns:
                df["price"] = pd.to_numeric(df[k].str.replace(",",""), errors="coerce")
    for col in df.columns:
        if "pnl" in col and "cumulative" not in col:
            df["pnl"] = pd.to_numeric(df[col].str.replace(",",""), errors="coerce")
    return df

def load_our(path):
    df = pd.read_csv(path, parse_dates=["entry_time","exit_time"], dtype=str)
    # coerce numeric columns
    for c in ("entry_price","exit_price","pnl"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_pairs_by_trade_number(tv):
    pairs = []
    if "trade_number" in tv.columns:
        for tn, g in tv.groupby("trade_number"):
            entry = None; exit_row = None
            for _, r in g.iterrows():
                typ = str(r.get("type","")).lower()
                sig = str(r.get("signal","")).lower()
                if "entry" in typ or "entry" in sig:
                    entry = r
                if "exit" in typ or "exit" in sig:
                    exit_row = r
            if entry is not None and exit_row is not None:
                pairs.append((entry, exit_row))
    else:
        i = 0
        n = len(tv)
        while i < n-1:
            r = tv.iloc[i]; r2 = tv.iloc[i+1]
            if ("entry" in str(r.get("type","")).lower()) or ("entry" in str(r.get("signal","")).lower()):
                pairs.append((r, r2)); i += 2
            else:
                i += 1
    return pairs

def canonical_side(s):
    s = str(s).lower()
    if "long" in s or "buy" in s: return "long"
    if "short" in s or "sell" in s: return "short"
    return ""

def float_eq(a,b,eps=1e-8):
    try:
        return abs(float(a)-float(b)) <= eps
    except Exception:
        return str(a).strip() == str(b).strip()

tv = load_tv(TV_PATH)
our = load_our(OUR_PATH)

print("TV rows:", len(tv), "Our trades:", len(our))
tv_pairs = build_pairs_by_trade_number(tv)
print("TV trade pairs found:", len(tv_pairs))

min_len = min(len(tv_pairs), len(our))
for idx in range(min_len):
    entry, exit_row = tv_pairs[idx]
    our_row = our.iloc[idx]
    tv_side = canonical_side(entry.get("signal") or entry.get("type"))
    tv_entry_price = entry.get("price")
    tv_exit_price = exit_row.get("price")
    tv_pnl = exit_row.get("pnl") if pd.notna(exit_row.get("pnl")) else entry.get("pnl")
    mismatches = []
    if tv_side != str(our_row.get("side","")).lower():
        mismatches.append(f"side tv={tv_side} our={our_row.get('side')}")
    if not float_eq(tv_entry_price, our_row.get("entry_price")):
        mismatches.append(f"entry_price tv={tv_entry_price} our={our_row.get('entry_price')}")
    if not float_eq(tv_exit_price, our_row.get("exit_price")):
        mismatches.append(f"exit_price tv={tv_exit_price} our={our_row.get('exit_price')}")
    if not float_eq(tv_pnl, our_row.get("pnl")):
        mismatches.append(f"pnl tv={tv_pnl} our={our_row.get('pnl')}")
    if mismatches:
        print("\nFirst mismatch at trade index", idx)
        print("TV entry row:"); print(entry.to_dict())
        print("TV exit row:"); print(exit_row.to_dict())
        print("Our trade row:"); print(our_row.to_dict())
        print("Mismatches:", mismatches)
        break
else:
    if len(tv_pairs) != len(our):
        print("No mismatch in first", min_len, "trades, but trade counts differ: TV", len(tv_pairs), "our", len(our))
    else:
        print("All compared trades match within tolerance.")
