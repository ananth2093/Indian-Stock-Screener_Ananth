# update_data.py
"""Headless data updater for the Indian screener.

Runs the legacy screener pipeline with a minimal Streamlit stub, then persists
Nifty 50 / Nifty 500 snapshots, a prediction archive, and price history.
"""

import importlib.util
import os
import sys
import types
import uuid
from datetime import datetime, time as _time, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


# ── Market-open throttle for the Pi scheduler ────────────────────────────────
def _is_in_market_open(dt: datetime | None = None) -> bool:
    """Rough Indian equity market-open check (IST, no holidays)."""
    from zoneinfo import ZoneInfo

    if dt is None:
        dt = datetime.now(ZoneInfo("Asia/Kolkata"))
    if dt.weekday() >= 5:
        return False
    return _time(9, 15) <= dt.time() < _time(15, 30)


def _should_update(data_dir: Path) -> bool:
    """Return True if we should run a full update now.

    Logic:
      - Always run if no snapshot exists yet.
      - Run every hour while the Indian market is open.
      - Run at most every 4 hours outside market hours.
    """
    last_update_file = data_dir / "last_update.json"
    now = datetime.now(timezone.utc)
    if not (data_dir / "latest_screener_nifty50.parquet").exists():
        return True
    if not (data_dir / "latest_screener_nifty500.parquet").exists():
        return True
    if not last_update_file.exists():
        return True
    try:
        last_update = datetime.fromisoformat(last_update_file.read_text().strip())
    except Exception:
        return True
    elapsed = (now - last_update).total_seconds()
    if _is_in_market_open():
        return elapsed >= 3500  # ~1 hour
    return elapsed >= 4 * 3600 - 100  # ~4 hours


# ── Minimal Streamlit stub ───────────────────────────────────────────────────
class _NoOp:
    """A do-nothing object that is also a context manager and callable."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, *args, **kwargs):
        return _NoOp()

    def __getattr__(self, name):
        return _NoOp()

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __getitem__(self, key):
        return _NoOp()


def _cache_data_decorator(fn):
    return fn


_cache_data_decorator.clear = lambda: None


def _make_columns(n, *args, **kwargs):
    count = len(n) if isinstance(n, (list, tuple)) else int(n)
    return tuple(_NoOp() for _ in range(count))


def _make_tabs(labels, *args, **kwargs):
    return [_NoOp() for _ in labels]


def _selectbox(label, options, *args, **kwargs):
    return (options[0] if options else None)


def _number_input(label, *args, **kwargs):
    return kwargs.get("value")


def _segmented_control(label, options, *args, **kwargs):
    return kwargs.get("default") or (options[0] if options else None)


def _button(*args, **kwargs):
    return False


def _stop():
    raise SystemExit(0)


def _build_streamlit_stub():
    st = types.ModuleType("streamlit")
    st.__dict__.update({
        "set_page_config": lambda *a, **k: None,
        "markdown": lambda *a, **k: None,
        "columns": _make_columns,
        "tabs": _make_tabs,
        "sidebar": _NoOp(),
        "spinner": lambda *a, **k: _NoOp(),
        "expander": lambda *a, **k: _NoOp(),
        "button": _button,
        "selectbox": _selectbox,
        "number_input": _number_input,
        "segmented_control": _segmented_control,
        "metric": lambda *a, **k: None,
        "info": lambda *a, **k: None,
        "warning": lambda *a, **k: None,
        "error": lambda *a, **k: None,
        "caption": lambda *a, **k: None,
        "success": lambda *a, **k: None,
        "dataframe": lambda *a, **k: None,
        "bar_chart": lambda *a, **k: None,
        "altair_chart": lambda *a, **k: None,
        "download_button": lambda *a, **k: None,
        "cache_data": lambda *a, **k: _cache_data_decorator,
        "rerun": lambda *a, **k: None,
        "stop": _stop,
        "session_state": {},
        "secrets": {},
        "empty": lambda *a, **k: _NoOp(),
        "progress": lambda *a, **k: _NoOp(),
        "html": lambda *a, **k: None,
    })

    def __getattr__(name):
        return _NoOp()

    st.__getattr__ = __getattr__
    return st


# ── Throttle check (Pi scheduler runs us every hour) ─────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
SCREENER_PATH = SCRIPT_DIR / "screener_app_legacy.py"

data_dir = SCRIPT_DIR / "data"
data_dir.mkdir(exist_ok=True)

if not _should_update(data_dir):
    print("Update skipped: outside market hours and recent update exists.")
    raise SystemExit(0)


# ── Run screener_app_legacy.py headlessly ──────────────────────────────────
sys.modules["streamlit"] = _build_streamlit_stub()

spec = importlib.util.spec_from_file_location("screener_app", SCREENER_PATH)
screener_mod = importlib.util.module_from_spec(spec)

# Force it to run its top-level (guarded) block as __main__.
screener_mod.__name__ = "__main__"

try:
    spec.loader.exec_module(screener_mod)
except SystemExit:
    pass

st_stub = sys.modules["streamlit"]
scr50 = st_stub.session_state.get("scr50")
scr500 = st_stub.session_state.get("scr500")

if scr50 is None or scr500 is None:
    raise RuntimeError(
        "screener_app_legacy.py did not produce `scr50` and/or `scr500`. "
        "Check the headless run output for errors."
    )

for _scr in (scr50, scr500):
    for col in _scr.columns:
        if _scr[col].dtype.name == "object":
            _scr[col] = _scr[col].infer_objects()

# ── Save latest screener snapshots ──────────────────────────────────────────
parquet_50 = data_dir / "latest_screener_nifty50.parquet"
csv_50 = data_dir / "latest_screener_nifty50.csv"
parquet_500 = data_dir / "latest_screener_nifty500.parquet"
csv_500 = data_dir / "latest_screener_nifty500.csv"

scr50.to_parquet(parquet_50, index=False)
scr50.to_csv(csv_50, index=False)
scr500.to_parquet(parquet_500, index=False)
scr500.to_csv(csv_500, index=False)

# ── Append to prediction archive ────────────────────────────────────────────
archive_path = data_dir / "prediction_archive.parquet"

run_timestamp = datetime.now(timezone.utc).isoformat()
run_id = str(uuid.uuid4())[:8]

archive_rows = []
for label, _scr in (("Nifty 50", scr50), ("Nifty 500", scr500)):
    if _scr is None or _scr.empty:
        continue
    cols = [c for c in _scr.columns if c != "Eligible"]
    tmp = _scr[cols].copy()
    tmp["index"] = label
    tmp["run_timestamp"] = run_timestamp
    tmp["run_id"] = run_id
    archive_rows.append(tmp)

if archive_rows:
    archive_row = pd.concat(archive_rows, ignore_index=True)
    if archive_path.exists():
        existing = pd.read_parquet(archive_path)
        archive_row = pd.concat([existing, archive_row], ignore_index=True)
    archive_row = archive_row.drop_duplicates(subset=["run_id", "Ticker", "index"])
    archive_row.to_parquet(archive_path, index=False)
else:
    archive_row = pd.DataFrame()

# ── Append to price history ─────────────────────────────────────────────────
price_history_path = data_dir / "price_history.parquet"

# Collect unique yfinance tickers from both snapshots.
yf_tickers = set()
for _scr in (scr50, scr500):
    if "YF Ticker" in _scr.columns:
        yf_tickers.update(_scr["YF Ticker"].dropna().astype(str).tolist())

yf_tickers = sorted(t for t in yf_tickers if t)

price_rows = []
if yf_tickers:
    try:
        # Bulk download latest close for all tickers in one shot.
        bulk = yf.download(
            tickers=" ".join(yf_tickers),
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if bulk is not None and not bulk.empty:
            if len(yf_tickers) == 1:
                # Single ticker returns a flat DataFrame.
                t = yf_tickers[0]
                if "Close" in bulk.columns:
                    close = bulk["Close"].dropna()
                    if not close.empty:
                        price_rows.append({
                            "date": pd.Timestamp(close.index[-1]).date(),
                            "ticker": t.upper(),
                            "close": float(close.iloc[-1]),
                        })
            else:
                for t in yf_tickers:
                    try:
                        sub = bulk[t]
                    except Exception:
                        continue
                    if isinstance(sub, pd.Series):
                        sub = sub.to_frame("Close")
                    if "Close" not in sub.columns:
                        continue
                    close = sub["Close"].dropna()
                    if close.empty:
                        continue
                    price_rows.append({
                        "date": pd.Timestamp(close.index[-1]).date(),
                        "ticker": t.upper(),
                        "close": float(close.iloc[-1]),
                    })
    except Exception as e:
        print(f"Bulk price download failed: {e}; falling back per ticker.")
        for t in yf_tickers:
            try:
                hist = yf.Ticker(t).history(period="5d", auto_adjust=True)
                if hist is None or hist.empty or "Close" not in hist.columns:
                    continue
                close = hist["Close"].dropna()
                price_rows.append({
                    "date": pd.Timestamp(close.index[-1]).date(),
                    "ticker": t.upper(),
                    "close": float(close.iloc[-1]),
                })
            except Exception:
                continue

# Add latest Nifty 50 index close (^NSEI).
for bench_ticker, bench_name in (("^NSEI", "NIFTY50"),):
    try:
        hist = yf.Ticker(bench_ticker).history(period="5d", auto_adjust=True)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            close = hist["Close"].dropna()
            price_rows.append({
                "date": pd.Timestamp(close.index[-1]).date(),
                "ticker": bench_name,
                "close": float(close.iloc[-1]),
            })
    except Exception:
        pass

price_history = pd.DataFrame(price_rows)
if price_history_path.exists():
    existing_ph = pd.read_parquet(price_history_path)
    price_history = pd.concat([existing_ph, price_history], ignore_index=True)

if not price_history.empty:
    price_history["date"] = pd.to_datetime(price_history["date"]).dt.date
    price_history = price_history.drop_duplicates(subset=["date", "ticker"])
    price_history.to_parquet(price_history_path, index=False)

# ── Mark successful update time ─────────────────────────────────────────────
(data_dir / "last_update.json").write_text(datetime.now(timezone.utc).isoformat())

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"Nifty 50 rows:     {len(scr50)}")
print(f"Nifty 500 rows:    {len(scr500)}")
print(f"Archive shape:     {archive_row.shape}")
print(f"Price rows:        {len(price_history)}")
print(f"  {parquet_50}")
print(f"  {parquet_500}")
