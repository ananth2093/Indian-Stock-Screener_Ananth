# app.py  (Nifty 50 Screener v4 — FMP primary, works on Streamlit Cloud)
# ─────────────────────────────────────────────────────────────────────────────
# v4 changes:
#   1. FMP as SOLE data source — works from any IP including Streamlit Cloud
#   2. Removed Yahoo Finance .info calls (rate-limited on cloud)
#   3. Removed NSE API calls (cookie-blocked on cloud)
#   4. FMP /quote        → price, MC, 52W, trailing PE
#   5. FMP /ratios-ttm   → Fwd PE, PEG, ROE, Op Margin, D/E, Int Coverage
#   6. FMP /income-statement (quarterly) → EPS, Earn Traj, Revenue
#   7. FMP /balance-sheet (quarterly)    → ROIC computation
#   8. yfinance batch download ONLY for momentum (price history — not blocked)
#   9. Universe: Wikipedia scrape (HTTP only — not blocked)
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random
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

# ── Constants ─────────────────────────────────────────────────────────────────
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
    "Financial Services":             "Financials",
    "Banking":                        "Financials",
    "Insurance":                      "Financials",
    "Diversified Financials":         "Financials",
    "Information Technology":         "Information Technology",
    "IT":                             "Information Technology",
    "Oil Gas & Consumable Fuels":     "Energy",
    "Oil & Gas":                      "Energy",
    "Energy":                         "Energy",
    "Power":                          "Utilities",
    "Utilities":                      "Utilities",
    "Fast Moving Consumer Goods":     "Consumer Staples",
    "FMCG":                           "Consumer Staples",
    "Consumer Goods":                 "Consumer Staples",
    "Tobacco":                        "Consumer Staples",
    "Automobile":                     "Consumer Discretionary",
    "Automobile And Auto Components": "Consumer Discretionary",
    "Consumer Durables":              "Consumer Discretionary",
    "Retailing":                      "Consumer Discretionary",
    "Construction":                   "Industrials",
    "Capital Goods":                  "Industrials",
    "Services":                       "Industrials",
    "Industrial Manufacturing":       "Industrials",
    "Infrastructure":                 "Industrials",
    "Ports & Shipping":               "Industrials",
    "Metals & Mining":                "Materials",
    "Metals":                         "Materials",
    "Mining":                         "Materials",
    "Cement & Cement Products":       "Materials",
    "Cement":                         "Materials",
    "Steel":                          "Materials",
    "Construction Materials":         "Materials",
    "Pharmaceuticals":                "Health Care",
    "Healthcare":                     "Health Care",
    "Pharma":                         "Health Care",
    "Hospital & Diagnostic Centres":  "Health Care",
    "Telecommunication":              "Communication Services",
    "Telecom":                        "Communication Services",
    "Media Entertainment & Publication": "Communication Services",
    "Real Estate":                    "Real Estate",
    "Realty":                         "Real Estate",
}

# ── FMP Key ───────────────────────────────────────────────────────────────────
def get_fmp_key():
    try:
        k = st.secrets["fmp"]["api_key"]
        return k if k and k.strip() and k != "YOUR_KEY_HERE" else None
    except Exception:
        return None

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    """Safe FMP API call with retry."""
    base = "https://financialmodelingprep.com/api/v3"
    p    = params or {}
    p["apikey"] = api_key
    try:
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]))
        session.mount("https://", adapter)
        r = session.get("{}/{}".format(base, endpoint), params=p, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def normalise_pct(val):
    if val is None: return None
    v = float(val)
    return v * 100.0 if abs(v) < 5.0 else v


# ── Universe (Wikipedia) ──────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_nifty50_universe():
    """Wikipedia scrape — reliable from cloud, no auth needed."""
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
                       for th in header_row.find_all(["th","td"])]
                      if header_row else [])
        ticker_col = next((i for i,h in enumerate(headers)
                           if any(k in h for k in ["symbol","ticker","nse"])), 2)
        sector_col = next((i for i,h in enumerate(headers)
                           if any(k in h for k in ["sector","industry","gics"])), 1)
        data = []
        for row in table.find_all("tr")[1:]:
            cols  = row.find_all(["td","th"])
            if len(cols) <= max(ticker_col, sector_col):
                continue
            raw_t = re.sub(r"$.*?$", "",
                           cols[ticker_col].get_text(strip=True)).strip()
            raw_t = re.sub(r"[^A-Za-z0-9&\-]", "", raw_t).upper()
            raw_s = re.sub(r"$.*?$", "",
                           cols[sector_col].get_text(strip=True)).strip()
            if not raw_t or len(raw_t) < 2:
                continue
            gics = SECTOR_MAP.get(raw_s)
            if gics is None:
                for nse_name, gics_name in SECTOR_MAP.items():
                    if (nse_name.lower() in raw_s.lower()
                            or raw_s.lower() in nse_name.lower()):
                        gics = gics_name; break
            data.append({"Ticker":     raw_t + ".NS",
                         "NSE Symbol": raw_t,
                         "Sector":     gics or raw_s,
                         "NSE Sector": raw_s})
        if len(data) < 30:
            raise RuntimeError("Only {} rows".format(len(data)))
        df = pd.DataFrame(data).drop_duplicates(subset=["Ticker"])
        st.success("✅ Universe: {} stocks from Wikipedia".format(len(df)))
        return df
    except Exception as e:
        st.warning("⚠️ Wikipedia failed: {}. Using fallback.".format(e))
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


# ── FMP Quote (price, MC, 52W, trailing PE) ───────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fmp_quotes(tickers, api_key):
    """
    FMP /quote/{symbols} — bulk endpoint, all 50 tickers in one call.
    Returns price, marketCap, 52W high/low, PE.
    """
    out = {t: {} for t in tickers}
    tl  = list(tickers)
    # FMP accepts comma-separated symbols
    chunks = [tl[i:i+50] for i in range(0, len(tl), 50)]
    for chunk in chunks:
        syms = ",".join(chunk)
        data = fmp_get("quote/{}".format(syms), api_key)
        if not data or not isinstance(data, list):
            continue
        for item in data:
            sym = str(item.get("symbol","")).upper().strip()
            if sym not in out:
                continue
            pe  = sf(item.get("pe"))
            mc  = sf(item.get("marketCap"))
            hi  = sf(item.get("yearHigh"))
            lo  = sf(item.get("yearLow"))
            px  = sf(item.get("price"))
            out[sym] = {
                "price": px,
                "mc":    mc,
                "hi52":  hi,
                "lo52":  lo,
                "pe":    pe if (pe and 0 < pe <= 10000) else None,
            }
        time.sleep(0.3)
    return out


# ── FMP Ratios TTM (Fwd PE, ROE, Op Margin, D/E, Int Coverage, PEG) ──────────
@st.cache_data(ttl=86400)
def fetch_fmp_ratios(tickers, api_key):
    """
    FMP /ratios-ttm/{symbol} — one call per ticker.
    Provides ROE, Op Margin, D/E, Int Coverage, ROIC.
    """
    out = {}
    tl  = list(tickers)

    def one(t):
        data = fmp_get("ratios-ttm/{}".format(t), api_key)
        if not data or not isinstance(data, list) or len(data) == 0:
            return t, {}
        item = data[0]
        peg_raw  = sf(item.get("priceEarningsGrowthRatioTTM"))
        roic_raw = sf(item.get("returnOnInvestedCapitalTTM"))
        roe_raw  = sf(item.get("returnOnEquityTTM"))
        om_raw   = sf(item.get("operatingProfitMarginTTM"))
        ic_raw   = sf(item.get("interestCoverageTTM"))
        de_raw   = sf(item.get("debtEquityRatioTTM"))
        fpe_raw  = sf(item.get("priceToEarningsRatioTTM"))
        return t, {
            "peg":          peg_raw if (peg_raw and 0 < peg_raw <= 500) else None,
            "peg_src":      "FMP" if (peg_raw and 0 < peg_raw <= 500) else None,
            "roic":         normalise_pct(roic_raw),
            "roe":          normalise_pct(roe_raw),
            "op_margin":    normalise_pct(om_raw),
            "int_coverage": min(float(ic_raw), 100.0) if (ic_raw and ic_raw > 0) else None,
            "debt_eq":      de_raw,
            "fwd_pe":       fpe_raw if (fpe_raw and 0 < fpe_raw <= 10000) else None,
        }

    CHUNK = 10; SLEEP = 1.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0)
    stat   = st.empty()
    for ci, chunk in enumerate(chunks):
        stat.text("FMP ratios: {}/{} tickers...".format(ci*CHUNK, len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for t, d in ex.map(one, chunk):
                out[t] = d
        prog.progress((ci+1)/len(chunks))
        if ci < len(chunks)-1:
            time.sleep(SLEEP)
    prog.empty(); stat.empty()
    return out


# ── FMP Income Statement (EPS, Earn Traj, Revenue) ───────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_income(tickers, api_key):
    """
    FMP /income-statement/{symbol}?period=quarter&limit=5
    Provides: revenue (4Q), EPS (trailing + forward proxy), earn traj
    """
    out = {}
    tl  = list(tickers)

    def one(t):
        data = fmp_get(
            "income-statement/{}".format(t), api_key,
            params={"period": "quarter", "limit": "5"}
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return t, {}
        # Most recent 4 quarters for revenue
        rev4 = []
        for q in data[:4]:
            rev4.append(sf(q.get("revenue")))
        # EPS trajectory: compare most recent vs one year ago
        eps_recent = sf(data[0].get("eps")) if len(data) > 0 else None
        eps_old    = sf(data[3].get("eps")) if len(data) >= 4 else None
        earn_traj  = None
        if eps_recent is not None and eps_old is not None and abs(eps_old) > 0.01:
            raw       = (eps_recent - eps_old) / abs(eps_old)
            earn_traj = max(-1.0, min(1.0, raw))
        # EPS growth YoY
        eps_growth = None
        if eps_recent is not None and eps_old is not None and abs(eps_old) > 0.01:
            eps_growth = (eps_recent - eps_old) / abs(eps_old) * 100.0
        return t, {
            "rev4":       rev4 if len(rev4) == 4 else [None]*4,
            "earn_traj":  earn_traj,
            "eps_growth": eps_growth,
        }

    CHUNK = 10; SLEEP = 1.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0)
    stat   = st.empty()
    for ci, chunk in enumerate(chunks):
        stat.text("FMP income statements: {}/{} tickers...".format(ci*CHUNK, len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for t, d in ex.map(one, chunk):
                out[t] = d
        prog.progress((ci+1)/len(chunks))
        if ci < len(chunks)-1:
            time.sleep(SLEEP)
    prog.empty(); stat.empty()
    return out


# ── FMP Balance Sheet (ROIC if not in ratios) ────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_balance_sheet(tickers, api_key):
    """
    FMP /balance-sheet-statement/{symbol}?period=quarter&limit=4
    Used only for ROIC computation fallback.
    """
    out = {}
    tl  = list(tickers)

    def one(t):
        data = fmp_get(
            "balance-sheet-statement/{}".format(t), api_key,
            params={"period": "quarter", "limit": "4"}
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return t, {}
        latest = data[0]
        return t, {
            "total_equity": sf(latest.get("totalStockholdersEquity")
                               or latest.get("totalEquity")),
            "total_debt":   sf(latest.get("totalDebt")
                               or latest.get("longTermDebt")),
            "cash":         sf(latest.get("cashAndCashEquivalents")
                               or latest.get("cashAndShortTermInvestments")),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        for t, d in ex.map(one, tl):
            out[t] = d
    return out


# ── Momentum (yfinance batch — price history only, not blocked) ───────────────
@st.cache_data(ttl=3600)
def fetch_momentum_batch(tickers):
    tl  = list(tickers)
    out = {t: {} for t in tl}
    try:
        raw_d = yf.download(tl, period="7mo", interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=True)
        raw_m = yf.download(tl, period="7mo", interval="1mo",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=True)
        for t in tl:
            try:
                closes_m = (raw_m["Close"].dropna() if len(tl) == 1
                            else raw_m[t]["Close"].dropna())
                closes_d = (raw_d["Close"].dropna() if len(tl) == 1
                            else raw_d[t]["Close"].dropna())
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
                skip = (r6-r1) if (r6 is not None and r1 is not None) else None
                mom  = None
                if skip is not None and trailing_vol and trailing_vol > 0:
                    mom = skip / trailing_vol
                elif skip is not None:
                    mom = skip
                out[t] = {"ret_1mo": r1, "ret_3mo": r3, "ret_6mo": r6,
                          "trailing_vol": trailing_vol, "momentum_score": mom}
            except Exception:
                pass
    except Exception:
        pass
    return out


# ── Quality Score ─────────────────────────────────────────────────────────────
def compute_quality_score(roic, roe, int_coverage, op_margin):
    scores = []
    prof = roic if roic is not None else roe
    if prof is not None and not pd.isna(prof):
        pf = float(prof)
        scores.append(min(100.0, np.log1p(pf)/np.log1p(30.0)*100.0) if pf > 0 else 0.0)
    else:
        scores.append(0.0)
    scores.append(min(100.0, max(0.0, float(int_coverage)/10.0*100.0))
                  if int_coverage is not None and not pd.isna(int_coverage) else 0.0)
    scores.append(min(100.0, max(0.0, float(op_margin)/40.0*100.0))
                  if op_margin is not None and not pd.isna(op_margin) else 0.0)
    return sum(scores) / 3.0


# ── Quality flag ──────────────────────────────────────────────────────────────
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


# ── Ranking ───────────────────────────────────────────────────────────────────
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
        elig["_s_quality"] = ((qs-q_min)/(q_max-q_min)*100.0
                              if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min
                              else qs.fillna(0.0))
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)
        raw = (W["valuation"]*elig["_s_val"] + W["quality"]*elig["_s_quality"]
               + W["peg"]*elig["_s_peg"]     + W["earn_traj"]*elig["_s_etraj"]
               + W["momentum"]*elig["_s_mom"])
        pen = elig.apply(lambda r: missing_factor_penalty(
            r, ["P/E","PEG","Quality Score","Earn Traj","Momentum Score"]), axis=1)
        elig["Score"] = raw * pen
        elig          = elig.sort_values("Score", ascending=False)
        elig["Rank"]  = range(1, len(elig)+1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]
    return scr


def compute_conviction_scores(scr):
    KEY = ["P/E","Fwd P/E","PEG","Quality Score","Momentum Score","Earn Traj"]
    scr = scr.copy()
    scr["_comp"] = scr.apply(
        lambda r: sum(1 for c in KEY if c in r.index and pd.notna(r[c]))/len(KEY),
        axis=1)
    med_pe  = scr["P/E"].median()
    sec_map = scr.groupby("Sector")["P/E"].median()
    def sec_disc(s):
        if pd.isna(med_pe) or med_pe == 0: return 1.0
        sp = sec_map.get(s)
        if pd.isna(sp) or sp == 0: return 1.0
        return float(np.clip(med_pe/sp, 0.7, 1.3))
    scr["_disc"] = scr["Sector"].map(sec_disc)
    raw  = scr["Score"] * scr["_comp"] * scr["_disc"]
    cmin, cmax = raw.min(), raw.max()
    scr["Conviction Score"] = ((raw-cmin)/(cmax-cmin)*100.0
                               if cmax > cmin else 50.0)
    return scr.drop(columns=["_comp","_disc"])


# ── Build table ───────────────────────────────────────────────────────────────
def build_screener_table(universe_df, fmp_quotes, fmp_ratios,
                         fmp_income, momentum_map):
    rows = []
    for _, r in universe_df.iterrows():
        t   = r["Ticker"]
        sec = r["Sector"]

        fq  = fmp_quotes.get(t, {})
        fr  = fmp_ratios.get(t, {})
        fi  = fmp_income.get(t, {})

        price     = to_num(fq.get("price"))
        mc        = to_num(fq.get("mc"))
        pe        = to_num(fq.get("pe"))
        hi        = to_num(fq.get("hi52"))
        lo        = to_num(fq.get("lo52"))
        fwd       = to_num(fr.get("fwd_pe"))
        roic      = to_num(fr.get("roic"))
        roe       = to_num(fr.get("roe"))
        ic        = to_num(fr.get("int_coverage"))
        om        = to_num(fr.get("op_margin"))
        de        = to_num(fr.get("debt_eq"))
        earn_traj = to_num(fi.get("earn_traj"))
        eps_g     = fi.get("eps_growth")

        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price-lo)/(hi-lo)*100.0)

        rev4                = fi.get("rev4", [None]*4)
        rq1,rq2,rq3,rq4    = [to_num(x) for x in rev4]
        growth              = revenue_growth_pct_cagr([rq1,rq2,rq3,rq4])

        # PEG
        peg_direct = to_num(fr.get("peg"))
        peg = None; peg_method = "—"
        if pd.notna(peg_direct):
            peg        = float(peg_direct)
            peg_method = "FMP"
        else:
            pe_for_peg = fwd if pd.notna(fwd) else pe
            if eps_g is not None:
                eg = float(eps_g)
                if eg >= MIN_GROWTH_PCT_FOR_PEG and pd.notna(pe_for_peg):
                    peg        = float(pe_for_peg) / eg
                    peg_method = "FMP EPS growth"
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
            "Ticker":             t.replace(".NS",""),
            "NSE Symbol":         t,
            "Sector":             sec,
            "Price (₹)":          price,
            "Mkt Cap (₹Cr)":      (mc/1e7) if mc is not None else None,
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
            "Rev Q1 (₹Cr)":       (rq1/1e7) if rq1 is not None else None,
            "Rev Q2 (₹Cr)":       (rq2/1e7) if rq2 is not None else None,
            "Rev Q3 (₹Cr)":       (rq3/1e7) if rq3 is not None else None,
            "Rev Q4 (₹Cr)":       (rq4/1e7) if rq4 is not None else None,
            "Rev Growth% (CAGR)": to_num(growth),
        })

    scr = pd.DataFrame(rows)
    if scr.empty: return scr

    total_mc = scr["Mkt Cap Raw"].sum()
    scr["MC% of Nifty50"] = (
        scr["Mkt Cap Raw"]/total_mc*100.0 if total_mc > 0 else None)

    num_cols = ["Price (₹)","Mkt Cap (₹Cr)","P/E","Fwd P/E","PEG",
                "52W Pos%","ROIC%","ROE%","Int Coverage","Op Margin%",
                "Debt/Eq","Quality Score","Earn Traj","Momentum Score",
                "Ret 1Mo%","Ret 3Mo%","Ret 6Mo%","Trailing Vol%",
                "MC% of Nifty50",
                "Rev Q1 (₹Cr)","Rev Q2 (₹Cr)","Rev Q3 (₹Cr)","Rev Q4 (₹Cr)",
                "Rev Growth% (CAGR)"]
    for c in num_cols:
        if c in scr.columns: scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns: scr["Rank"] = pd.NA
    scr = compute_conviction_scores(scr)
    return scr


# ── KPI panel ─────────────────────────────────────────────────────────────────
def render_sector_kpi_panel(scr, sector_sel):
    def _kpi(label, value, sub, color="#ffffff"):
        return ("<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;"
                "text-align:center;margin:2px;'>"
                "<div style='color:#aaa;font-size:11px;margin-bottom:4px;'>{}</div>"
                "<div style='color:{};font-size:20px;font-weight:700;'>{}</div>"
                "<div style='color:#666;font-size:10px;margin-top:3px;'>{}</div>"
                "</div>").format(label,color,value,sub)

    is_all   = (sector_sel == "All Sectors")
    label    = "All Sectors (Nifty 50)" if is_all else sector_sel
    total_mc = scr["Mkt Cap Raw"].sum()
    sdata    = scr.copy() if is_all else scr[scr["Sector"]==sector_sel]
    sec_mc   = sdata["Mkt Cap Raw"].sum()
    pct      = (100.0 if is_all
                else (sec_mc/total_mc*100.0 if total_mc > 0 else 0.0))
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

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.markdown(_kpi("Sector Mkt Cap",   fmt_mc_inr(sec_mc),  "sector total"),         unsafe_allow_html=True)
    c2.markdown(_kpi("Nifty 50 Mkt Cap", fmt_mc_inr(total_mc),"all 50 stocks"),        unsafe_allow_html=True)
    c3.markdown(_kpi("Sector Share",     "{:.1f}%".format(pct),"{} stocks".format(len(sdata))), unsafe_allow_html=True)
    c4.markdown(_kpi("Median P/E → Fwd",
                     "{:.1f}→{:.1f}".format(med_pe,med_fwd)
                     if pd.notna(med_pe) and pd.notna(med_fwd) else "N/A",
                     "trailing → forward","#facc15"), unsafe_allow_html=True)
    c5.markdown(_kpi("Median Quality",
                     "{:.0f}/100".format(med_qual) if pd.notna(med_qual) else "N/A",
                     "ROIC+IntCov+Margin","#4ade80"), unsafe_allow_html=True)
    c6.markdown(_kpi("Median PEG",
                     "{:.2f}".format(med_peg) if pd.notna(med_peg) else "N/A",
                     "price/earnings/growth","#a78bfa"), unsafe_allow_html=True)

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
# APP ENTRY POINT
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

page_screener, page_about, page_debug = st.tabs(["📊 Screener","📖 About","🔧 Debug"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with page_screener:
    fmp_key = get_fmp_key()
    if not fmp_key:
        st.error("❌ FMP API key not configured. Add [fmp] api_key to Streamlit Secrets.")
        st.markdown("""
        **How to add your FMP key:**
        1. Get a free key at [financialmodelingprep.com](https://financialmodelingprep.com)
        2. In Streamlit Cloud → your app → **Settings** → **Secrets**
        3. Add:
        ```toml
        [fmp]
        api_key = "your_key_here"
        ```
        4. Save — app restarts automatically
        """)
        st.stop()

    col_r, col_t = st.columns([1,6])
    with col_r:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_t:
        st.caption("Last loaded: {} · Prices: 1hr · Fundamentals: 24hr".format(
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

    with st.spinner("Fetching momentum (price history)..."):
        momentum = fetch_momentum_batch(tickers)

    # Coverage banner
    total_t   = len(tickers)
    has_price = sum(1 for t in tickers if fmp_quotes.get(t,{}).get("price") is not None)
    has_pe    = sum(1 for t in tickers if fmp_quotes.get(t,{}).get("pe")    is not None)
    has_fwd   = sum(1 for t in tickers if fmp_ratios.get(t,{}).get("fwd_pe")    is not None)
    has_roe   = sum(1 for t in tickers if fmp_ratios.get(t,{}).get("roe")       is not None)
    has_roic  = sum(1 for t in tickers if fmp_ratios.get(t,{}).get("roic")      is not None)
    has_et    = sum(1 for t in tickers if fmp_income.get(t,{}).get("earn_traj") is not None)

    st.info(
        "Data coverage — "
        "Price: {}/{} ({:.0f}%) · "
        "P/E: {}/{} ({:.0f}%) · "
        "Fwd P/E: {}/{} ({:.0f}%) · "
        "ROE: {}/{} ({:.0f}%) · "
        "ROIC: {}/{} ({:.0f}%) · "
        "Earn Traj: {}/{} ({:.0f}%) · "
        "Source: FMP API".format(
            has_price, total_t, has_price/total_t*100,
            has_pe,    total_t, has_pe   /total_t*100,
            has_fwd,   total_t, has_fwd  /total_t*100,
            has_roe,   total_t, has_roe  /total_t*100,
            has_roic,  total_t, has_roic /total_t*100,
            has_et,    total_t, has_et   /total_t*100,
        )
    )

    with st.spinner("Building screener table..."):
        scr = build_screener_table(
            universe_df, fmp_quotes, fmp_ratios, fmp_income, momentum)

    # Filters
    st.markdown("### Filters")
    with st.expander("Valuation & Size", expanded=True):
        fc1,fc2,fc3,fc4,fc5 = st.columns(5)
        all_sectors = sorted(scr["Sector"].dropna().unique().tolist())
        sector_sel  = fc1.selectbox("Sector", ["All Sectors"]+all_sectors)
        sort_by     = fc2.selectbox("Sort by", [
            "Sector then Rank","Score high to low","Conviction high to low",
            "MC% of Nifty50 high to low",
            "Price low to high","Price high to low","Mkt Cap high to low",
            "PE low to high","Fwd PE low to high","PEG low to high",
            "Quality Score high","ROIC high to low","Earn Traj high to low",
            "Momentum Score high","52W Pos low to high","Rev Growth high to low",
        ])
        pe_max   = fc3.number_input("Max PE",            value=9999,  step=10)
        peg_max  = fc4.number_input("Max PEG",           value=999.0, step=1.0)
        mc_min_c = fc5.number_input("Min Mkt Cap (₹Cr)", value=0,     step=5000)

    with st.expander("Quality Filters", expanded=False):
        qc1,qc2,qc3,qc4 = st.columns(4)
        roic_min_f = qc1.number_input("Min ROIC (%)",         value=0.0, step=5.0)
        ic_min_f   = qc2.number_input("Min Int Coverage (x)", value=0.0, step=1.0)
        om_min_f   = qc3.number_input("Min Op Margin (%)",    value=0.0, step=5.0)
        qual_min_f = qc4.number_input("Min Quality Score",    value=0.0, step=5.0)

    with st.expander("Momentum & Earnings", expanded=False):
        mc1,mc2 = st.columns(2)
        mom_min = mc1.number_input("Min Momentum Score", value=-999.0, step=5.0)
        et_min  = mc2.number_input("Min Earn Traj",      value=-1.0,   step=0.1)

    render_sector_kpi_panel(scr, sector_sel)

    filt = scr.copy()
    if sector_sel != "All Sectors":
        filt = filt[filt["Sector"]==sector_sel]
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
        "Sector then Rank":          (["Sector","Rank"],      [True,True]),
        "Score high to low":         (["Score"],              [False]),
        "Conviction high to low":    (["Conviction Score"],   [False]),
        "MC% of Nifty50 high to low":(["MC% of Nifty50"],    [False]),
        "Price low to high":         (["Price (₹)"],          [True]),
        "Price high to low":         (["Price (₹)"],          [False]),
        "Mkt Cap high to low":       (["Mkt Cap (₹Cr)"],      [False]),
        "PE low to high":            (["P/E"],                [True]),
        "Fwd PE low to high":        (["Fwd P/E"],            [True]),
        "PEG low to high":           (["PEG"],                [True]),
        "Quality Score high":        (["Quality Score"],      [False]),
        "ROIC high to low":          (["ROIC%"],              [False]),
        "Earn Traj high to low":     (["Earn Traj"],          [False]),
        "Momentum Score high":       (["Momentum Score"],     [False]),
        "52W Pos low to high":       (["52W Pos%"],           [True]),
        "Rev Growth high to low":    (["Rev Growth% (CAGR)"], [False]),
    }
    sc,sa = sort_map.get(sort_by,(["Sector","Rank"],[True,True]))
    filt  = filt.sort_values(sc, ascending=sa, na_position="last")

    st.caption("Showing {} of {} stocks · Sector: {} · Sort: {}".format(
        len(filt), len(scr), sector_sel, sort_by))

    disp = filt.copy()
    for c in ["P/E","Fwd P/E","PEG","Earn Traj","52W Pos%",
              "ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
              "Quality Score","Momentum Score","Ret 1Mo%","Ret 3Mo%",
              "Ret 6Mo%","Trailing Vol%","Score","Conviction Score",
              "Rev Growth% (CAGR)","MC% of Nifty50",
              "Price (₹)","Mkt Cap (₹Cr)",
              "Rev Q1 (₹Cr)","Rev Q2 (₹Cr)","Rev Q3 (₹Cr)","Rev Q4 (₹Cr)"]:
        if c in disp.columns: disp[c] = disp[c].round(2)

    disp["Quality Flag"] = disp.apply(
        lambda r: quality_flag(r.get("ROIC%"),r.get("ROE%"),
                               r.get("Int Coverage"),
                               r.get("Op Margin%"),r.get("Debt/Eq")), axis=1)
    disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

    COLS = [
        "Ticker","Sector",
        "Price (₹)","Mkt Cap (₹Cr)","MC% of Nifty50",
        "P/E","Fwd P/E","PEG","PEG Method",
        "Earn Traj",
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
        file_name="nifty50_screener_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
        mime="text/csv",
    )
    st.markdown("""
**PEG:** < 1.0 = potentially undervalued. Only computed when EPS growth ≥ 5%.

**Earn Traj:** Year-over-year EPS change from FMP quarterly income statements. Range −1.0 to +1.0.

**MC% of Nifty50:** This stock's share of total Nifty 50 market cap.
""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with page_about:
    st.markdown("## About — Nifty 50 Screener v4")
    st.markdown("""
### Data Architecture
| Field | Source | Endpoint |
|-------|--------|----------|
| Price, MC, 52W, Trailing P/E | **FMP** | `/quote/{symbol}` |
| Fwd P/E, PEG, ROE, Op Margin, D/E, Int Coverage, ROIC | **FMP** | `/ratios-ttm/{symbol}` |
| EPS, Revenue (quarterly), Earn Traj | **FMP** | `/income-statement/{symbol}?period=quarter` |
| Momentum (price returns, volatility) | **yfinance batch** | price history download |
| Universe | **Wikipedia** | NIFTY_50 page |

### Why FMP?
Yahoo Finance and NSE India API both rate-limit Streamlit Cloud server IPs.
FMP provides an authenticated API that works from any IP.
Free tier = 250 calls/day. Nifty 50 screener uses ~150 calls/day total.

### Scoring Model
`Valuation 25% + Quality 25% + PEG 20% + Earn Traj 15% + Momentum 15%`
""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DEBUG
# ══════════════════════════════════════════════════════════════════════════════
with page_debug:
    st.markdown("## 🔧 Debug")
    fmp_key_dbg = get_fmp_key()
    test_sym    = st.text_input("Ticker (with .NS)", value="RELIANCE.NS")

    if st.button("▶ Run diagnostic"):
        if not fmp_key_dbg:
            st.error("No FMP key configured in Streamlit Secrets")
        else:
            with st.spinner("Testing {}...".format(test_sym)):

                st.markdown("### FMP /quote")
                d = fmp_get("quote/{}".format(test_sym), fmp_key_dbg)
                if d and isinstance(d, list) and len(d) > 0:
                    st.success("✅ FMP quote working")
                    item = d[0]
                    st.json({k: item.get(k) for k in
                             ["price","pe","marketCap","yearHigh","yearLow","name"]})
                else:
                    st.error("❌ FMP quote returned empty — check API key or ticker format")

                st.markdown("### FMP /ratios-ttm")
                d = fmp_get("ratios-ttm/{}".format(test_sym), fmp_key_dbg)
                if d and isinstance(d, list) and len(d) > 0:
                    st.success("✅ FMP ratios working")
                    item = d[0]
                    st.json({k: item.get(k) for k in [
                        "returnOnEquityTTM","returnOnInvestedCapitalTTM",
                        "operatingProfitMarginTTM","interestCoverageTTM",
                        "priceEarningsGrowthRatioTTM","priceToEarningsRatioTTM",
                    ]})
                else:
                    st.error("❌ FMP ratios-ttm empty — may need paid tier")

                st.markdown("### FMP /income-statement (quarterly)")
                d = fmp_get("income-statement/{}".format(test_sym), fmp_key_dbg,
                            params={"period":"quarter","limit":"4"})
                if d and isinstance(d, list) and len(d) > 0:
                    st.success("✅ FMP income statement working — {} quarters".format(len(d)))
                    st.json({k: d[0].get(k) for k in
                             ["date","revenue","eps","netIncome"]})
                else:
                    st.error("❌ FMP income statement empty")

                st.markdown("### yfinance version")
                st.code(yf.__version__)
