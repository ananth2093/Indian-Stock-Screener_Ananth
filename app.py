# app.py  (Nifty 50 Screener v5 — FMP primary, debugged)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import re
import warnings
import concurrent.futures
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("pip install beautifulsoup4")
    st.stop()

MIN_GROWTH_PCT_FOR_PEG = 5.0

FACTOR_WEIGHTS = {
    "valuation":  0.25,
    "quality":    0.25,
    "peg":        0.20,
    "earn_traj":  0.15,
    "momentum":   0.15,
}

QUALITY_THRESHOLDS = {
    "roic_min":         8.0,
    "int_coverage_min": 3.0,
    "op_margin_min":    5.0,
}

SECTOR_MAP = {
    "Financial Services":                "Financials",
    "Banking":                           "Financials",
    "Insurance":                         "Financials",
    "Diversified Financials":            "Financials",
    "Information Technology":            "Information Technology",
    "IT":                                "Information Technology",
    "Oil Gas & Consumable Fuels":        "Energy",
    "Oil & Gas":                         "Energy",
    "Energy":                            "Energy",
    "Power":                             "Utilities",
    "Utilities":                         "Utilities",
    "Fast Moving Consumer Goods":        "Consumer Staples",
    "FMCG":                              "Consumer Staples",
    "Consumer Goods":                    "Consumer Staples",
    "Tobacco":                           "Consumer Staples",
    "Automobile":                        "Consumer Discretionary",
    "Automobile And Auto Components":    "Consumer Discretionary",
    "Consumer Durables":                 "Consumer Discretionary",
    "Retailing":                         "Consumer Discretionary",
    "Construction":                      "Industrials",
    "Capital Goods":                     "Industrials",
    "Services":                          "Industrials",
    "Industrial Manufacturing":          "Industrials",
    "Infrastructure":                    "Industrials",
    "Ports & Shipping":                  "Industrials",
    "Metals & Mining":                   "Materials",
    "Metals":                            "Materials",
    "Mining":                            "Materials",
    "Cement & Cement Products":          "Materials",
    "Cement":                            "Materials",
    "Steel":                             "Materials",
    "Construction Materials":            "Materials",
    "Pharmaceuticals":                   "Health Care",
    "Healthcare":                        "Health Care",
    "Pharma":                            "Health Care",
    "Hospital & Diagnostic Centres":     "Health Care",
    "Telecommunication":                 "Communication Services",
    "Telecom":                           "Communication Services",
    "Media Entertainment & Publication": "Communication Services",
    "Real Estate":                       "Real Estate",
    "Realty":                            "Real Estate",
}

# ─── FMP Key ──────────────────────────────────────────────────────────────────
def get_fmp_key():
    try:
        k = st.secrets["fmp"]["api_key"]
        return k if k and k.strip() and k != "YOUR_KEY_HERE" else None
    except Exception:
        return None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def sf(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

def fmt_mc_inr(val):
    if pd.isna(val) or val == 0:
        return "N/A"
    cr = val / 1e7
    if cr >= 100000:
        return "₹{:.2f}L Cr".format(cr / 100000)
    return "₹{:.0f}Cr".format(cr)

def percentile_score(series: pd.Series, ascending=True) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    valid  = series.notna()
    if valid.sum() == 0:
        return result.fillna(0.0)
    ranked = series[valid].rank(method="average", ascending=ascending)
    n      = valid.sum()
    result[valid]  = (ranked - 1) / (n - 1) * 100.0 if n > 1 else 50.0
    result[~valid] = 0.0
    return result

def missing_factor_penalty(row, factor_cols):
    missing = sum(1 for c in factor_cols if pd.isna(row.get(c)))
    if missing >= 3: return 0.70
    if missing == 2: return 0.85
    return 1.0

def revenue_growth_pct_cagr(rev4):
    try:
        if rev4 is None or len(rev4) != 4:
            return None
        q1, _, _, q4 = rev4
        if q1 is None or q4 is None:
            return None
        q1, q4 = float(q1), float(q4)
        if q1 <= 0 or q4 <= 0:
            return None
        return ((q4 / q1) ** (1 / 3) - 1) * 100.0
    except Exception:
        return None

def fmp_get(endpoint, api_key, params=None):
    base = "https://financialmodelingprep.com/api/v3"
    p    = dict(params or {})
    p["apikey"] = api_key
    try:
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]))
        session.mount("https://", adapter)
        r = session.get("{}/{}".format(base, endpoint), params=p, timeout=20)
        if r.status_code == 403:
            return None
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "Error Message" in data:
            return None
        return data
    except Exception:
        return None

def decimal_to_pct(val):
    """Convert FMP decimal ratios (0.15) → percentage (15.0)."""
    if val is None:
        return None
    v = float(val)
    if abs(v) <= 20.0:
        return v * 100.0
    return v

# ─── Universe ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_nifty50_universe():
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/NIFTY_50",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )
        r.raise_for_status()
        soup  = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            for tbl in soup.find_all("table", {"class": "wikitable"}):
                hdrs = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
                if any("symbol" in h or "ticker" in h for h in hdrs):
                    table = tbl; break
        if table is None:
            table = soup.find("table", {"class": "wikitable sortable"})
        if table is None:
            for tbl in soup.find_all("table", {"class": "wikitable"}):
                if len(tbl.find_all("tr")) >= 30:
                    table = tbl; break
        if table is None:
            raise RuntimeError("Table not found")

        header_row = table.find("tr")
        headers    = ([th.get_text(strip=True).lower()
                       for th in header_row.find_all(["th", "td"])]
                      if header_row else [])
        ticker_col = next((i for i, h in enumerate(headers)
                           if any(k in h for k in ["symbol","ticker","nse"])), 2)
        sector_col = next((i for i, h in enumerate(headers)
                           if any(k in h for k in ["sector","industry","gics"])), 1)
        data = []
        for row in table.find_all("tr")[1:]:
            cols  = row.find_all(["td", "th"])
            if len(cols) <= max(ticker_col, sector_col):
                continue
            raw_t = re.sub(r"$.*?$", "", cols[ticker_col].get_text(strip=True)).strip()
            raw_t = re.sub(r"[^A-Za-z0-9&\-]", "", raw_t).upper()
            raw_s = re.sub(r"$.*?$", "", cols[sector_col].get_text(strip=True)).strip()
            if not raw_t or len(raw_t) < 2:
                continue
            gics = SECTOR_MAP.get(raw_s)
            if gics is None:
                for nse_name, gics_name in SECTOR_MAP.items():
                    if (nse_name.lower() in raw_s.lower()
                            or raw_s.lower() in nse_name.lower()):
                        gics = gics_name; break
            data.append({
                "Ticker":     raw_t + ".NS",
                "NSE Symbol": raw_t,
                "Sector":     gics or raw_s,
                "NSE Sector": raw_s,
            })
        if len(data) < 30:
            raise RuntimeError("Only {} rows".format(len(data)))
        df = pd.DataFrame(data).drop_duplicates(subset=["Ticker"])
        st.success("✅ Universe: {} stocks from Wikipedia".format(len(df)))
        return df
    except Exception as e:
        st.warning("⚠️ Wikipedia failed: {}. Using fallback list.".format(e))
        return pd.DataFrame([
            {"Ticker":"RELIANCE.NS",   "NSE Symbol":"RELIANCE",   "Sector":"Energy",                 "NSE Sector":"Oil Gas & Consumable Fuels"},
            {"Ticker":"TCS.NS",        "NSE Symbol":"TCS",        "Sector":"Information Technology", "NSE Sector":"Information Technology"},
            {"Ticker":"HDFCBANK.NS",   "NSE Symbol":"HDFCBANK",   "Sector":"Financials",             "NSE Sector":"Financial Services"},
            {"Ticker":"INFY.NS",       "NSE Symbol":"INFY",       "Sector":"Information Technology", "NSE Sector":"Information Technology"},
            {"Ticker":"ICICIBANK.NS",  "NSE Symbol":"ICICIBANK",  "Sector":"Financials",             "NSE Sector":"Financial Services"},
            {"Ticker":"HINDUNILVR.NS", "NSE Symbol":"HINDUNILVR", "Sector":"Consumer Staples",       "NSE Sector":"FMCG"},
            {"Ticker":"ITC.NS",        "NSE Symbol":"ITC",        "Sector":"Consumer Staples",       "NSE Sector":"FMCG"},
            {"Ticker":"SBIN.NS",       "NSE Symbol":"SBIN",       "Sector":"Financials",             "NSE Sector":"Financial Services"},
            {"Ticker":"BHARTIARTL.NS", "NSE Symbol":"BHARTIARTL", "Sector":"Communication Services", "NSE Sector":"Telecommunication"},
            {"Ticker":"LT.NS",         "NSE Symbol":"LT",         "Sector":"Industrials",            "NSE Sector":"Construction"},
        ])

# ─── FMP Quotes ───────────────────────────────────────────────────────────────
def _fill_quote(out, sym, item):
    pe = sf(item.get("pe"))
    mc = sf(item.get("marketCap"))
    hi = sf(item.get("yearHigh"))
    lo = sf(item.get("yearLow"))
    px = sf(item.get("price"))
    out[sym] = {
        "price": px,
        "mc":    mc,
        "hi52":  hi,
        "lo52":  lo,
        "pe":    pe if (pe is not None and 0 < pe <= 10000) else None,
    }

@st.cache_data(ttl=3600)
def fetch_fmp_quotes(tickers, api_key):
    out = {t: {} for t in tickers}
    tl  = list(tickers)

    # Attempt 1: bulk call
    syms_str = ",".join(tl)
    data = fmp_get("quote/{}".format(syms_str), api_key)
    if data and isinstance(data, list) and len(data) > 0:
        for item in data:
            sym = str(item.get("symbol", "")).upper().strip()
            if sym in out:
                _fill_quote(out, sym, item)
        filled = sum(1 for v in out.values() if v.get("price") is not None)
        if filled >= len(tl) * 0.5:
            return out

    # Attempt 2: individual fallback for missing
    missing = [t for t in tl if out[t].get("price") is None]
    if missing:
        def one_quote(t):
            d = fmp_get("quote/{}".format(t), api_key)
            if d and isinstance(d, list) and len(d) > 0:
                return t, d[0]
            return t, None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for t, item in ex.map(one_quote, missing):
                if item:
                    _fill_quote(out, t, item)
        time.sleep(0.5)
    return out

# ─── FMP Ratios TTM ───────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_ratios(tickers, api_key):
    out = {}
    tl  = list(tickers)

    def one(t):
        data = fmp_get("ratios-ttm/{}".format(t), api_key)
        if not data or not isinstance(data, list) or len(data) == 0:
            return t, {}
        item     = data[0]
        peg_raw  = sf(item.get("priceEarningsGrowthRatioTTM"))
        roic_raw = sf(item.get("returnOnInvestedCapitalTTM"))
        roe_raw  = sf(item.get("returnOnEquityTTM"))
        om_raw   = sf(item.get("operatingProfitMarginTTM"))
        ic_raw   = sf(item.get("interestCoverageTTM"))
        de_raw   = sf(item.get("debtEquityRatioTTM"))
        pe_ttm   = sf(item.get("priceToEarningsRatioTTM"))
        roic     = decimal_to_pct(roic_raw)
        roe      = decimal_to_pct(roe_raw)
        om       = decimal_to_pct(om_raw)
        ic       = float(ic_raw) if (ic_raw is not None and ic_raw > 0) else None
        if ic is not None:
            ic = min(ic, 100.0)
        peg      = float(peg_raw) if (peg_raw is not None and 0 < peg_raw <= 500) else None
        fwd_pe   = float(pe_ttm)  if (pe_ttm  is not None and 0 < pe_ttm  <= 10000) else None
        return t, {
            "peg":          peg,
            "roic":         roic,
            "roe":          roe,
            "op_margin":    om,
            "int_coverage": ic,
            "debt_eq":      de_raw,
            "fwd_pe":       fwd_pe,
        }

    CHUNK = 10; SLEEP = 1.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0)
    stat   = st.empty()
    for ci, chunk in enumerate(chunks):
        stat.text("FMP ratios: {}/{} tickers...".format(ci * CHUNK, len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for t, d in ex.map(one, chunk):
                out[t] = d
        prog.progress((ci + 1) / len(chunks))
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
    prog.empty(); stat.empty()
    return out

# ─── FMP Income Statement ─────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_income(tickers, api_key):
    out = {}
    tl  = list(tickers)

    def one(t):
        data = fmp_get(
            "income-statement/{}".format(t), api_key,
            params={"period": "quarter", "limit": "5"}
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return t, {}
        rev4 = [sf(q.get("revenue")) for q in data[:4]]
        while len(rev4) < 4:
            rev4.append(None)
        eps_recent = sf(data[0].get("eps")) if len(data) > 0 else None
        eps_old    = sf(data[3].get("eps")) if len(data) >= 4 else None
        earn_traj  = None
        eps_growth = None
        if (eps_recent is not None and eps_old is not None
                and abs(eps_old) > 0.001):
            raw        = (eps_recent - eps_old) / abs(eps_old)
            earn_traj  = max(-1.0, min(1.0, raw))
            eps_growth = raw * 100.0
        return t, {
            "rev4":       rev4,
            "earn_traj":  earn_traj,
            "eps_growth": eps_growth,
        }

    CHUNK = 10; SLEEP = 1.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0)
    stat   = st.empty()
    for ci, chunk in enumerate(chunks):
        stat.text("FMP income: {}/{} tickers...".format(ci * CHUNK, len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for t, d in ex.map(one, chunk):
                out[t] = d
        prog.progress((ci + 1) / len(chunks))
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
    prog.empty(); stat.empty()
    return out

# ─── Momentum ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_momentum_batch(tickers):
    tl  = list(tickers)
    out = {t: {} for t in tl}

    def _get_close(df, ticker):
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.Index) and "Close" in df.columns:
            return df["Close"].dropna()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                return df["Close"][ticker].dropna()
            except (KeyError, TypeError):
                pass
        try:
            return df[ticker]["Close"].dropna()
        except (KeyError, TypeError):
            pass
        return pd.Series(dtype=float)

    try:
        if len(tl) == 1:
            raw_d = yf.download(tl[0], period="7mo", interval="1d",
                                auto_adjust=True, progress=False)
            raw_m = yf.download(tl[0], period="7mo", interval="1mo",
                                auto_adjust=True, progress=False)
        else:
            raw_d = yf.download(tl, period="7mo", interval="1d",
                                group_by="ticker", auto_adjust=True,
                                progress=False, threads=True)
            raw_m = yf.download(tl, period="7mo", interval="1mo",
                                group_by="ticker", auto_adjust=True,
                                progress=False, threads=True)
    except Exception:
        return out

    for t in tl:
        try:
            closes_m = _get_close(raw_m, t)
            closes_d = _get_close(raw_d, t)
            if len(closes_m) < 2:
                continue
            px_now = float(closes_m.iloc[-1])

            def ret_mo(n):
                idx = -(n + 1)
                if abs(idx) > len(closes_m): return None
                px = float(closes_m.iloc[idx])
                return (px_now / px - 1) * 100.0 if px > 0 else None

            r1 = ret_mo(1); r3 = ret_mo(3); r6 = ret_mo(6)
            trailing_vol = None
            if len(closes_d) >= 20:
                dr = closes_d.pct_change().dropna().tail(90)
                if len(dr) >= 15:
                    trailing_vol = float(dr.std() * np.sqrt(252) * 100.0)
            skip = (r6 - r1) if (r6 is not None and r1 is not None) else None
            mom  = None
            if skip is not None and trailing_vol and trailing_vol > 0:
                mom = skip / trailing_vol
            elif skip is not None:
                mom = skip
            out[t] = {
                "ret_1mo": r1, "ret_3mo": r3, "ret_6mo": r6,
                "trailing_vol": trailing_vol, "momentum_score": mom,
            }
        except Exception:
            pass
    return out

# ─── Quality ──────────────────────────────────────────────────────────────────
def compute_quality_score(roic, roe, int_coverage, op_margin):
    scores = []
    prof = roic if roic is not None else roe
    if prof is not None and not pd.isna(prof):
        pf = float(prof)
        scores.append(
            min(100.0, np.log1p(max(pf, 0)) / np.log1p(30.0) * 100.0)
            if pf > 0 else 0.0)
    else:
        scores.append(0.0)
    scores.append(
        min(100.0, max(0.0, float(int_coverage) / 10.0 * 100.0))
        if int_coverage is not None and not pd.isna(int_coverage) else 0.0)
    scores.append(
        min(100.0, max(0.0, float(op_margin) / 40.0 * 100.0))
        if op_margin is not None and not pd.isna(op_margin) else 0.0)
    return sum(scores) / 3.0

def quality_flag(roic, roe, ic, om, de):
    flags = []
    prof  = roic if (roic is not None and not pd.isna(roic)) else roe
    if prof is not None and not pd.isna(prof) and prof < QUALITY_THRESHOLDS["roic_min"]:
        flags.append("ROIC<8%" if (roic is not None and not pd.isna(roic)) else "ROE<8%")
    if ic is not None and not pd.isna(ic) and ic < QUALITY_THRESHOLDS["int_coverage_min"]:
        flags.append("IntCov<3x")
    if om is not None and not pd.isna(om) and om < QUALITY_THRESHOLDS["op_margin_min"]:
        flags.append("Margin<5%")
    de_note = " | D/E:{:.1f}".format(de) if (de is not None and not pd.isna(de)) else ""
    return (", ".join(flags) if flags else "Pass") + de_note

# ─── Ranking ──────────────────────────────────────────────────────────────────
def compute_rank_by_sector(scr):
    scr = scr.copy()
    scr["Score"] = pd.NA
    scr["Rank"]  = pd.NA
    W = FACTOR_WEIGHTS
    for sector in scr["Sector"].dropna().unique():
        elig = scr[(scr["Sector"] == sector) & scr["Eligible"]].copy()
        if elig.empty: continue
        pe_input         = elig["Fwd P/E"].fillna(elig["P/E"])
        elig["_s_val"]   = percentile_score(pe_input,               ascending=True)
        elig["_s_peg"]   = percentile_score(elig["PEG"],            ascending=True)
        elig["_s_mom"]   = percentile_score(elig["Momentum Score"], ascending=False)
        elig["_s_etraj"] = percentile_score(elig["Earn Traj"],      ascending=False)
        qs = elig["Quality Score"]
        q_min, q_max = qs.min(), qs.max()
        elig["_s_quality"] = (
            (qs - q_min) / (q_max - q_min) * 100.0
            if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min
            else qs.fillna(0.0))
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)
        raw = (W["valuation"] * elig["_s_val"]   + W["quality"]   * elig["_s_quality"]
             + W["peg"]       * elig["_s_peg"]   + W["earn_traj"] * elig["_s_etraj"]
             + W["momentum"]  * elig["_s_mom"])
        pen = elig.apply(lambda r: missing_factor_penalty(
            r, ["P/E","PEG","Quality Score","Earn Traj","Momentum Score"]), axis=1)
        elig["Score"] = raw * pen
        elig          = elig.sort_values("Score", ascending=False)
        elig["Rank"]  = range(1, len(elig) + 1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]
    return scr

def compute_conviction_scores(scr):
    KEY  = ["P/E","Fwd P/E","PEG","Quality Score","Momentum Score","Earn Traj"]
    scr  = scr.copy()
    scr["_comp"] = scr.apply(
        lambda r: sum(1 for c in KEY if c in r.index and pd.notna(r[c])) / len(KEY),
        axis=1)
    med_pe  = scr["P/E"].median()
    sec_map = scr.groupby("Sector")["P/E"].median()
    def sec_disc(s):
        if pd.isna(med_pe) or med_pe == 0: return 1.0
        sp = sec_map.get(s)
        if pd.isna(sp) or sp == 0: return 1.0
        return float(np.clip(med_pe / sp, 0.7, 1.3))
    scr["_disc"] = scr["Sector"].map(sec_disc)
    raw  = scr["Score"] * scr["_comp"] * scr["_disc"]
    cmin, cmax = raw.min(), raw.max()
    scr["Conviction Score"] = (
        (raw - cmin) / (cmax - cmin) * 100.0 if cmax > cmin else 50.0)
    return scr.drop(columns=["_comp","_disc"])

# ─── Build Table ──────────────────────────────────────────────────────────────
def build_screener_table(universe_df, fmp_quotes, fmp_ratios, fmp_income, momentum_map):
    rows = []
    for _, r in universe_df.iterrows():
        t   = r["Ticker"]
        sec = r["Sector"]
        fq  = fmp_quotes.get(t, {})
        fr  = fmp_ratios.get(t, {})
        fi  = fmp_income.get(t, {})

        price     = to_num(fq.get("price"))
        mc        = to_num(fq.get("mc"))
        pe_quote  = to_num(fq.get("pe"))
        pe_ratios = to_num(fr.get("fwd_pe"))
        pe        = pe_quote if pd.notna(pe_quote) else pe_ratios
        hi        = to_num(fq.get("hi52"))
        lo        = to_num(fq.get("lo52"))
        fwd       = (pe_ratios
                     if (pd.notna(pe_ratios) and
                         (pd.isna(pe_quote) or
                          abs(float(pe_ratios or 0) - float(pe_quote or 0)) > 0.5))
                     else None)
        roic      = to_num(fr.get("roic"))
        roe       = to_num(fr.get("roe"))
        ic        = to_num(fr.get("int_coverage"))
        om        = to_num(fr.get("op_margin"))
        de        = to_num(fr.get("debt_eq"))
        earn_traj = to_num(fi.get("earn_traj"))
        eps_g     = fi.get("eps_growth")

        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price - lo) / (hi - lo) * 100.0)

        rev4             = fi.get("rev4", [None]*4)
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth = revenue_growth_pct_cagr([rq1, rq2, rq3, rq4])

        peg_direct = to_num(fr.get("peg"))
        peg = None; peg_method = "—"
        if pd.notna(peg_direct):
            peg = float(peg_direct); peg_method = "FMP"
        else:
            pe_for_peg = fwd if pd.notna(fwd) else pe
            if eps_g is not None and pd.notna(pe_for_peg):
                eg = float(eps_g)
                if eg >= MIN_GROWTH_PCT_FOR_PEG:
                    peg = float(pe_for_peg) / eg; peg_method = "Calc (EPS g)"
        if peg is not None and (peg <= 0 or peg > 500):
            peg = None

        q_score = compute_quality_score(
            float(roic) if pd.notna(roic) else None,
            float(roe)  if pd.notna(roe)  else None,
            float(ic)   if pd.notna(ic)   else None,
            float(om)   if pd.notna(om)   else None,
        )

        mom       = momentum_map.get(t, {})
        ret_1mo   = to_num(mom.get("ret_1mo"))
        ret_3mo   = to_num(mom.get("ret_3mo"))
        ret_6mo   = to_num(mom.get("ret_6mo"))
        mom_score = to_num(mom.get("momentum_score"))
        t_vol     = to_num(mom.get("trailing_vol"))

        rows.append({
            "Ticker":             t.replace(".NS", ""),
            "NSE Symbol":         t,
            "Sector":             sec,
            "Price (₹)":          price,
            "Mkt Cap (₹Cr)":      (mc / 1e7) if mc is not None else None,
            "Mkt Cap Raw":        mc,
            "P/E":                pe,
            "Fwd P/E":            fwd,
            "PEG":                to_num(peg),
            "PEG Method":         peg_method,
            "Earn Traj":          earn_traj,
            "52W Pos%":           to_num(pos52),
            "ROIC%":              roic,
            "ROE%":               roe,
            "Int Coverage":       ic,
            "Op Margin%":         om,
            "Debt/Eq":            de,
            "Quality Score":      to_num(q_score),
            "Momentum Score":     mom_score,
            "Ret 1Mo%":           ret_1mo,
            "Ret 3Mo%":           ret_3mo,
            "Ret 6Mo%":           ret_6mo,
            "Trailing Vol%":      t_vol,
            "Eligible":           True,
            "Rev Q1 (₹Cr)":       (rq1 / 1e7) if rq1 is not None else None,
            "Rev Q2 (₹Cr)":       (rq2 / 1e7) if rq2 is not None else None,
            "Rev Q3 (₹Cr)":       (rq3 / 1e7) if rq3 is not None else None,
            "Rev Q4 (₹Cr)":       (rq4 / 1e7) if rq4 is not None else None,
            "Rev Growth% (CAGR)": to_num(growth),
        })

    scr = pd.DataFrame(rows)
    if scr.empty: return scr

    total_mc = scr["Mkt Cap Raw"].sum()
    scr["MC% of Nifty50"] = (
        scr["Mkt Cap Raw"] / total_mc * 100.0 if total_mc > 0 else None)

    num_cols = [
        "Price (₹)","Mkt Cap (₹Cr)","P/E","Fwd P/E","PEG","52W Pos%",
        "ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
        "Quality Score","Earn Traj","Momentum Score",
        "Ret 1Mo%","Ret 3Mo%","Ret 6Mo%","Trailing Vol%","MC% of Nifty50",
        "Rev Q1 (₹Cr)","Rev Q2 (₹Cr)","Rev Q3 (₹Cr)","Rev Q4 (₹Cr)",
        "Rev Growth% (CAGR)",
    ]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns: scr["Rank"] = pd.NA
    scr = compute_conviction_scores(scr)
    return scr

# ─── KPI Panel ────────────────────────────────────────────────────────────────
def render_sector_kpi_panel(scr, sector_sel):
    def _kpi(label, value, sub, color="#ffffff"):
        return (
            "<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;"
            "text-align:center;margin:2px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:4px;'>{}</div>"
            "<div style='color:{};font-size:20px;font-weight:700;'>{}</div>"
            "<div style='color:#666;font-size:10px;margin-top:3px;'>{}</div>"
            "</div>"
        ).format(label, color, value, sub)

    is_all   = (sector_sel == "All Sectors")
    label    = "All Sectors (Nifty 50)" if is_all else sector_sel
    total_mc = scr["Mkt Cap Raw"].sum()
    sdata    = scr.copy() if is_all else scr[scr["Sector"] == sector_sel]
    sec_mc   = sdata["Mkt Cap Raw"].sum()
    pct      = (100.0 if is_all
                else (sec_mc / total_mc * 100.0 if total_mc > 0 else 0.0))
    med_pe   = sdata["P/E"].median()
    med_fwd  = sdata["Fwd P/E"].median()
    med_qual = sdata["Quality Score"].median()
    med_peg  = sdata["PEG"].median()

    st.markdown(
        "<div style='background:#12122a;border:1px solid #2a2a4a;border-radius:12px;"
        "padding:16px 20px;margin-bottom:16px;'>"
        "<span style='color:#aaa;font-size:13px;'>Sector Analysis  </span>"
        "<span style='color:#fff;font-size:14px;font-weight:700;'>{}</span>"
        "</div>".format(label), unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(_kpi("Sector Mkt Cap",   fmt_mc_inr(sec_mc),  "sector total"),               unsafe_allow_html=True)
    c2.markdown(_kpi("Nifty 50 Mkt Cap", fmt_mc_inr(total_mc),"all 50 stocks"),              unsafe_allow_html=True)
    c3.markdown(_kpi("Sector Share",     "{:.1f}%".format(pct), "{} stocks".format(len(sdata))), unsafe_allow_html=True)
    c4.markdown(_kpi("Median P/E → Fwd",
                     "{:.1f}→{:.1f}".format(med_pe, med_fwd)
                     if pd.notna(med_pe) and pd.notna(med_fwd) else
                     ("{:.1f}".format(med_pe) if pd.notna(med_pe) else "N/A"),
                     "trailing → forward", "#facc15"), unsafe_allow_html=True)
    c5.markdown(_kpi("Median Quality",
                     "{:.0f}/100".format(med_qual) if pd.notna(med_qual) else "N/A",
                     "ROIC+IntCov+Margin", "#4ade80"), unsafe_allow_html=True)
    c6.markdown(_kpi("Median PEG",
                     "{:.2f}".format(med_peg) if pd.notna(med_peg) else "N/A",
                     "price/earnings/growth", "#a78bfa"), unsafe_allow_html=True)

    if not is_all:
        top3   = sdata[sdata["Rank"].notna()].sort_values("Rank").head(3)
        badges = "  ".join(
            "<span style='background:#1a2a4a;color:#93c5fd;padding:3px 10px;"
            "border-radius:6px;font-weight:700;font-size:13px;'>{} "
            "<span style='color:#4ade80;font-size:11px;'>#{}</span></span>".format(
                row["Ticker"], int(row["Rank"]))
            for _, row in top3.iterrows())
        st.markdown(
            "<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;"
            "margin-bottom:12px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>Top Ranked in Sector</div>"
            "<div>{}</div></div>".format(
                badges or "<span style='color:#555;'>No ranked stocks</span>"),
            unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nifty 50 Screener",
    layout="wide",
    page_icon="🇮🇳",
    initial_sidebar_state="collapsed",
)
st.markdown(
    "<style>div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}</style>",
    unsafe_allow_html=True)

st.markdown("## 🇮🇳 Nifty 50 Fundamental Screener")
st.caption("FMP API · Wikipedia universe · 5-factor scoring · INR")

page_screener, page_about, page_debug = st.tabs(["📊 Screener", "📖 About", "🔧 Debug"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with page_screener:
    fmp_key = get_fmp_key()
    if not fmp_key:
        st.error("❌ FMP API key not configured.")
        st.markdown("""
**How to add your FMP key:**
1. Get a free key at [financialmodelingprep.com](https://financialmodelingprep.com)
2. Streamlit Cloud → your app → **Settings** → **Secrets**
3. Add:
```toml
[fmp]
api_key = "your_key_here"
""")
st.stop()

col_r, col_t = st.columns([1, 6])
with col_r:
if st.button("🔄 Refresh"):
st.cache_data.clear()
st.rerun()
with col_t:
st.caption("Last loaded: {} · Prices: 1hr cache · Fundamentals: 24hr cache".format(
datetime.now().strftime("%I:%M %p")))

with st.spinner("Loading universe from Wikipedia..."):
universe_df = get_nifty50_universe()
tickers = tuple(universe_df["Ticker"].tolist())

with st.spinner("Fetching FMP quotes (price, PE, MC, 52W)..."):
fmp_quotes = fetch_fmp_quotes(tickers, fmp_key)

with st.spinner("Fetching FMP ratios (ROE, margins, ROIC, PEG)..."):
fmp_ratios = fetch_fmp_ratios(tickers, fmp_key)

with st.spinner("Fetching FMP income statements (EPS, revenue)..."):
fmp_income = fetch_fmp_income(tickers, fmp_key)

with st.spinner("Fetching momentum data (yfinance)..."):
momentum = fetch_momentum_batch(tickers)

total_t   = len(tickers)
has_price = sum(1 for t in tickers if fmp_quotes.get(t, {}).get("price") is not None)
has_pe    = sum(1 for t in tickers if fmp_quotes.get(t, {}).get("pe")    is not None)
has_roe   = sum(1 for t in tickers if fmp_ratios.get(t, {}).get("roe")   is not None)
has_roic  = sum(1 for t in tickers if fmp_ratios.get(t, {}).get("roic")  is not None)
has_et    = sum(1 for t in tickers if fmp_income.get(t, {}).get("earn_traj") is not None)
has_mom   = sum(1 for t in tickers if momentum.get(t, {}).get("momentum_score") is not None)

coverage_color = "info" if has_price >= total_t * 0.7 else "warning"
getattr(st, coverage_color)(
"Data coverage — "
"Price: {}/{} ({:.0f}%) · "
"P/E: {}/{} ({:.0f}%) · "
"ROE: {}/{} ({:.0f}%) · "
"ROIC: {}/{} ({:.0f}%) · "
"Earn Traj: {}/{} ({:.0f}%) · "
"Momentum: {}/{} ({:.0f}%) · "
"Source: FMP + yfinance".format(
has_price, total_t, has_price / total_t * 100,
has_pe,    total_t, has_pe    / total_t * 100,
has_roe,   total_t, has_roe   / total_t * 100,
has_roic,  total_t, has_roic  / total_t * 100,
has_et,    total_t, has_et    / total_t * 100,
has_mom,   total_t, has_mom   / total_t * 100,
)
)

with st.spinner("Building screener table..."):
scr = build_screener_table(
universe_df, fmp_quotes, fmp_ratios, fmp_income, momentum)

if scr.empty:
st.error("No data returned. Check your FMP API key in the Debug tab.")
st.stop()

#── Filters ───────────────────────────────────────────────────────────────
st.markdown("### Filters")
with st.expander("Valuation & Size", expanded=True):
fc1, fc2, fc3, fc4, fc5 = st.columns(5)
all_sectors = sorted(scr["Sector"].dropna().unique().tolist())
sector_sel  = fc1.selectbox("Sector", ["All Sectors"] + all_sectors)
sort_by     = fc2.selectbox("Sort by", [
"Sector then Rank","Score high to low","Conviction high to low",
"MC% of Nifty50 high to low","Price low to high","Price high to low",
"Mkt Cap high to low","PE low to high","Fwd PE low to high",
"PEG low to high","Quality Score high","ROIC high to low",
"Earn Traj high to low","Momentum Score high",
"52W Pos low to high","Rev Growth high to low",
])
pe_max   = fc3.number_input("Max PE",            value=9999,  step=10)
peg_max  = fc4.number_input("Max PEG",           value=999.0, step=1.0)
mc_min_c = fc5.number_input("Min Mkt Cap (₹Cr)", value=0,     step=5000)

with st.expander("Quality Filters", expanded=False):
qc1, qc2, qc3, qc4 = st.columns(4)
roic_min_f = qc1.number_input("Min ROIC (%)",         value=0.0, step=5.0)
ic_min_f   = qc2.number_input("Min Int Coverage (x)", value=0.0, step=1.0)
om_min_f   = qc3.number_input("Min Op Margin (%)",    value=0.0, step=5.0)
qual_min_f = qc4.number_input("Min Quality Score",    value=0.0, step=5.0)

with st.expander("Momentum & Earnings", expanded=False):
mc1, mc2 = st.columns(2)
mom_min = mc1.number_input("Min Momentum Score", value=-999.0, step=5.0)
et_min  = mc2.number_input("Min Earn Traj",      value=-1.0,   step=0.1)

render_sector_kpi_panel(scr, sector_sel)

filt = scr.copy()
if sector_sel != "All Sectors":
filt = filt[filt["Sector"] == sector_sel]
filt = filt[(filt["Mkt Cap (₹Cr)"].isna()) | (filt["Mkt Cap (₹Cr)"] >= mc_min_c)]
filt = filt[(filt["P/E"].isna())            | (filt["P/E"]           <= pe_max)]
filt = filt[(filt["PEG"].isna())            | (filt["PEG"]           <= peg_max)]
filt = filt[(filt["ROIC%"].isna())          | (filt["ROIC%"]         >= roic_min_f)]
filt = filt[(filt["Int Coverage"].isna())   | (filt["Int Coverage"]  >= ic_min_f)]
filt = filt[(filt["Op Margin%"].isna())     | (filt["Op Margin%"]    >= om_min_f)]
filt = filt[(filt["Quality Score"].isna())  | (filt["Quality Score"] >= qual_min_f)]
filt = filt[(filt["Momentum Score"].isna()) | (filt["Momentum Score"] >= mom_min)]
filt = filt[(filt["Earn Traj"].isna())      | (filt["Earn Traj"]     >= et_min)]

sort_map = {
"Sector then Rank":           (["Sector","Rank"],       [True, True]),
"Score high to low":          (["Score"],               [False]),
"Conviction high to low":     (["Conviction Score"],    [False]),
"MC% of Nifty50 high to low": (["MC% of Nifty50"],     [False]),
"Price low to high":          (["Price (₹)"],           [True]),
"Price high to low":          (["Price (₹)"],           [False]),
"Mkt Cap high to low":        (["Mkt Cap (₹Cr)"],       [False]),
"PE low to high":             (["P/E"],                 [True]),
"Fwd PE low to high":         (["Fwd P/E"],             [True]),
"PEG low to high":            (["PEG"],                 [True]),
"Quality Score high":         (["Quality Score"],       [False]),
"ROIC high to low":           (["ROIC%"],               [False]),
"Earn Traj high to low":      (["Earn Traj"],           [False]),
"Momentum Score high":        (["Momentum Score"],      [False]),
"52W Pos low to high":        (["52W Pos%"],            [True]),
"Rev Growth high to low":     (["Rev Growth% (CAGR)"],  [False]),
}
sc, sa = sort_map.get(sort_by, (["Sector","Rank"], [True, True]))
filt   = filt.sort_values(sc, ascending=sa, na_position="last")

st.caption("Showing {} of {} stocks · Sector: {} · Sort: {}".format(
len(filt), len(scr), sector_sel, sort_by))

disp = filt.copy()
for c in ["P/E","Fwd P/E","PEG","Earn Traj","52W Pos%",
"ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
"Quality Score","Momentum Score","Ret 1Mo%","Ret 3Mo%",
"Ret 6Mo%","Trailing Vol%","Score","Conviction Score",
"Rev Growth% (CAGR)","MC% of Nifty50","Price (₹)","Mkt Cap (₹Cr)",
"Rev Q1 (₹Cr)","Rev Q2 (₹Cr)","Rev Q3 (₹Cr)","Rev Q4 (₹Cr)"]:
if c in disp.columns:
disp[c] = disp[c].round(2)

disp["Quality Flag"] = disp.apply(
lambda r: quality_flag(r.get("ROIC%"), r.get("ROE%"),
r.get("Int Coverage"),
r.get("Op Margin%"), r.get("Debt/Eq")), axis=1)
disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

COLS = [
"Ticker","Sector",
"Price (₹)","Mkt Cap (₹Cr)","MC% of Nifty50",
"P/E","Fwd P/E","PEG","PEG Method","Earn Traj",
"ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
"Quality Score","Quality Flag",
"Momentum Score","Ret 1Mo%","Ret 3Mo%","Ret 6Mo%","Trailing Vol%",
"52W Pos%","Score","Conviction Score","Rank",
"Rev Q1 (₹Cr)","Rev Q2 (₹Cr)","Rev Q3 (₹Cr)","Rev Q4 (₹Cr)",
"Rev Growth% (CAGR)",
]
disp_final = disp[[c for c in COLS if c in disp.columns]].copy()
st.dataframe(disp_final, use_container_width=True, height=680)

st.download_button(
label="⬇ Download CSV",
data=disp_final.to_csv(index=False).encode("utf-8"),
file_name="nifty50_screener_{}.csv".format(
datetime.now().strftime("%Y%m%d_%H%M")),
mime="text/csv",
)

st.markdown("""
P/E: Trailing twelve months from FMP /quote.

Fwd P/E: TTM P/E from FMP /ratios-ttm (true forward PE requires FMP paid tier).

PEG: < 1.0 = potentially undervalued. Only computed when EPS growth ≥ 5%.

Earn Traj: YoY EPS change from quarterly income statements. Range −1.0 to +1.0.

ROIC / ROE / Op Margin: Converted from FMP decimal format (0.15 → 15%).

MC% of Nifty50: Stock's share of total Nifty 50 market cap.
""")

══════════════════════════════════════════════════════════════════════════════
TAB 2 — ABOUT
══════════════════════════════════════════════════════════════════════════════
with page_about:
st.markdown("## About — Nifty 50 Screener v5")
st.markdown("""

Data Architecture
Field	Source	Endpoint
Price, MC, 52W, Trailing P/E	FMP	/quote/{symbol} (bulk)
TTM P/E, PEG, ROE, Op Margin, D/E, Int Coverage, ROIC	FMP	/ratios-ttm/{symbol}
EPS, Revenue (quarterly), Earn Traj	FMP	/income-statement?period=quarter
Momentum (price returns, volatility)	yfinance	price history download
Universe	Wikipedia	NIFTY_50 page
FMP Free Tier Limitations
Available Free	Requires Paid Tier
Trailing P/E	True Forward P/E
TTM Ratios (ROE, ROIC, PEG)	Analyst estimates
Quarterly income / balance sheet	Real-time quotes
Scoring Model
Valuation 25% + Quality 25% + PEG 20% + Earn Traj 15% + Momentum 15%

Scores are sector-relative percentile ranks. Missing data applies a penalty
(−15% for 2 missing factors, −30% for 3 or more).
""")

══════════════════════════════════════════════════════════════════════════════
TAB 3 — DEBUG
══════════════════════════════════════════════════════════════════════════════
with page_debug:
st.markdown("## 🔧 Debug — API Diagnostics")
fmp_key_dbg = get_fmp_key()
test_sym    = st.text_input("Ticker (with .NS)", value="RELIANCE.NS")
if st.button("▶ Run diagnostic"):
    if not fmp_key_dbg:
        st.error("No FMP key found in Streamlit Secrets.")
    else:
        with st.spinner("Testing {}...".format(test_sym)):

            st.markdown("### 1. FMP /quote")
            d = fmp_get("quote/{}".format(test_sym), fmp_key_dbg)
            if d and isinstance(d, list) and len(d) > 0:
                item = d[0]
                st.success("✅ /quote OK")
                st.json({k: item.get(k) for k in
                         ["symbol","price","pe","marketCap",
                          "yearHigh","yearLow","name","currency"]})
            else:
                st.error("❌ /quote empty — verify API key and that ticker "
                         "format is RELIANCE.NS not RELIANCE")

            st.markdown("### 2. FMP /ratios-ttm")
            d = fmp_get("ratios-ttm/{}".format(test_sym), fmp_key_dbg)
            if d and isinstance(d, list) and len(d) > 0:
                item = d[0]
                st.success("✅ /ratios-ttm OK")
                st.json({k: item.get(k) for k in [
                    "returnOnEquityTTM",
                    "returnOnInvestedCapitalTTM",
                    "operatingProfitMarginTTM",
                    "interestCoverageTTM",
                    "priceEarningsGrowthRatioTTM",
                    "priceToEarningsRatioTTM",
                    "debtEquityRatioTTM",
                ]})
            else:
                st.warning("⚠️ /ratios-ttm empty — may require FMP paid tier")

            st.markdown("### 3. FMP /income-statement (quarterly)")
            d = fmp_get(
                "income-statement/{}".format(test_sym), fmp_key_dbg,
                params={"period": "quarter", "limit": "4"}
            )
            if d and isinstance(d, list) and len(d) > 0:
                st.success("✅ /income-statement OK — {} quarters".format(len(d)))
                st.json({k: d[0].get(k) for k in
                         ["date","revenue","eps","netIncome","period"]})
            else:
                st.error("❌ /income-statement empty")

            st.markdown("### 4. yfinance momentum test")
            try:
                test_df = yf.download(
                    test_sym, period="3mo", interval="1d",
                    auto_adjust=True, progress=False)
                if not test_df.empty:
                    st.success("✅ yfinance OK — {} rows, latest close: {:.2f}".format(
                        len(test_df),
                        float(test_df["Close"].dropna().iloc[-1])))
                else:
                    st.warning("⚠️ yfinance returned empty DataFrame")
            except Exception as ex:
                st.error("❌ yfinance error: {}".format(ex))

            st.markdown("### 5. API key check")
            st.code("Key prefix: {}...  Length: {}".format(
                fmp_key_dbg[:6] if len(fmp_key_dbg) > 6 else fmp_key_dbg,
                len(fmp_key_dbg)))
