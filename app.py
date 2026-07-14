# app.py  (Nifty Screener v8 — simplified filters + comparison fixes)
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
import math

warnings.filterwarnings("ignore")

try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("pip install beautifulsoup4")
    st.stop()

# ── Canonical column names ────────────────────────────────────────────────────
COL_MC  = "Mkt Cap (LCr)"
COL_RQ1 = "Rev Q1 (1000Cr)"
COL_RQ2 = "Rev Q2 (1000Cr)"
COL_RQ3 = "Rev Q3 (1000Cr)"
COL_RQ4 = "Rev Q4 (1000Cr)"

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

# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def sf(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

def fmt_mc_inr(val):
    if val is None or (isinstance(val, float) and pd.isna(val)) or val == 0:
        return "N/A"
    return "Rs.{:.2f}L Cr".format(val / 1e12)

def percentile_score(series, ascending=True):
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

def revenue_growth_yoy(rev4):
    try:
        if rev4 is None or len(rev4) != 4: return None
        q_newest, q_oldest = rev4[0], rev4[3]
        if q_newest is None or q_oldest is None: return None
        q_newest, q_oldest = float(q_newest), float(q_oldest)
        if q_newest <= 0 or q_oldest <= 0: return None
        return (q_newest / q_oldest - 1) * 100.0
    except Exception:
        return None

def decimal_to_pct(val):
    if val is None: return None
    v = float(val)
    return v * 100.0 if abs(v) <= 20.0 else v

def safe_float(obj):
    if obj is None: return None
    if isinstance(obj, pd.Series):
        obj = obj.dropna()
        if obj.empty: return None
        obj = obj.iloc[0]
    try:
        f = float(obj)
        return None if np.isnan(f) else f
    except Exception:
        return None

def _extract_scalar(info, *keys, default=None):
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                f = float(v)
                if not np.isnan(f): return f
            except Exception:
                pass
    return default

# ─── Universe helpers ─────────────────────────────────────────────────────────
def _parse_wiki_table(soup, min_rows=30):
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
            if len(tbl.find_all("tr")) >= min_rows:
                table = tbl; break
    return table

def _extract_rows(table):
    header_row = table.find("tr")
    headers = (
        [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]
        if header_row else []
    )
    ticker_col = next(
        (i for i, h in enumerate(headers) if any(k in h for k in ["symbol","ticker","nse"])), 2)
    sector_col = next(
        (i for i, h in enumerate(headers) if any(k in h for k in ["sector","industry","gics"])), 1)
    data = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all(["td","th"])
        if len(cols) <= max(ticker_col, sector_col): continue
        raw_t = re.sub(r"$.*?$", "", cols[ticker_col].get_text(strip=True)).strip()
        raw_t = re.sub(r"[^A-Za-z0-9&\-]", "", raw_t).upper()
        raw_s = re.sub(r"$.*?$", "", cols[sector_col].get_text(strip=True)).strip()
        if not raw_t or len(raw_t) < 2: continue
        gics = SECTOR_MAP.get(raw_s)
        if gics is None:
            for nse_name, gics_name in SECTOR_MAP.items():
                if nse_name.lower() in raw_s.lower() or raw_s.lower() in nse_name.lower():
                    gics = gics_name; break
        data.append({"Base": raw_t, "Ticker": raw_t+".NS",
                     "Sector": gics or raw_s, "NSE Sector": raw_s})
    return data

@st.cache_data(ttl=86400)
def get_nifty50_universe():
    try:
        r = requests.get("https://en.wikipedia.org/wiki/NIFTY_50",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        soup  = BeautifulSoup(r.text, "html.parser")
        table = _parse_wiki_table(soup, min_rows=30)
        if table is None: raise RuntimeError("Table not found")
        data = _extract_rows(table)
        if len(data) < 30: raise RuntimeError("Only {} rows".format(len(data)))
        return pd.DataFrame(data).drop_duplicates(subset=["Ticker"])
    except Exception as e:
        st.warning("Nifty 50 Wikipedia failed: {}. Using fallback.".format(e))
        fallback = [
            ("RELIANCE","Energy"),("TCS","Information Technology"),
            ("HDFCBANK","Financials"),("INFY","Information Technology"),
            ("ICICIBANK","Financials"),("HINDUNILVR","Consumer Staples"),
            ("ITC","Consumer Staples"),("SBIN","Financials"),
            ("BHARTIARTL","Communication Services"),("LT","Industrials"),
            ("KOTAKBANK","Financials"),("AXISBANK","Financials"),
            ("WIPRO","Information Technology"),("HCLTECH","Information Technology"),
            ("ASIANPAINT","Materials"),("MARUTI","Consumer Discretionary"),
            ("BAJFINANCE","Financials"),("TITAN","Consumer Discretionary"),
            ("SUNPHARMA","Health Care"),("ULTRACEMCO","Materials"),
        ]
        return pd.DataFrame([
            {"Base":b,"Ticker":b+".NS","Sector":s,"NSE Sector":s} for b,s in fallback])

@st.cache_data(ttl=86400)
def get_nifty500_universe():
    try:
        r = requests.get("https://en.wikipedia.org/wiki/NIFTY_500",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        soup  = BeautifulSoup(r.text, "html.parser")
        table = _parse_wiki_table(soup, min_rows=50)
        if table is None: raise RuntimeError("Table not found")
        data = _extract_rows(table)
        if len(data) < 50: raise RuntimeError("Only {} rows".format(len(data)))
        return pd.DataFrame(data).drop_duplicates(subset=["Ticker"])
    except Exception as e:
        st.warning("Nifty 500 Wikipedia failed: {}. Using extended fallback.".format(e))
        fallback = [
            ("RELIANCE","Energy"),("TCS","Information Technology"),("HDFCBANK","Financials"),
            ("INFY","Information Technology"),("ICICIBANK","Financials"),
            ("HINDUNILVR","Consumer Staples"),("ITC","Consumer Staples"),
            ("SBIN","Financials"),("BHARTIARTL","Communication Services"),
            ("LT","Industrials"),("KOTAKBANK","Financials"),("AXISBANK","Financials"),
            ("WIPRO","Information Technology"),("HCLTECH","Information Technology"),
            ("ASIANPAINT","Materials"),("MARUTI","Consumer Discretionary"),
            ("BAJFINANCE","Financials"),("TITAN","Consumer Discretionary"),
            ("SUNPHARMA","Health Care"),("ULTRACEMCO","Materials"),
            ("NESTLEIND","Consumer Staples"),("POWERGRID","Utilities"),
            ("NTPC","Utilities"),("ONGC","Energy"),("COALINDIA","Energy"),
            ("JSWSTEEL","Materials"),("TATASTEEL","Materials"),("HINDALCO","Materials"),
            ("GRASIM","Materials"),("ADANIENT","Industrials"),("ADANIPORTS","Industrials"),
            ("DIVISLAB","Health Care"),("DRREDDY","Health Care"),("CIPLA","Health Care"),
            ("APOLLOHOSP","Health Care"),("TORNTPHARM","Health Care"),
            ("TECHM","Information Technology"),("MPHASIS","Information Technology"),
            ("PERSISTENT","Information Technology"),("LTIM","Information Technology"),
            ("INDUSINDBK","Financials"),("FEDERALBNK","Financials"),
            ("BANDHANBNK","Financials"),("IDFCFIRSTB","Financials"),
            ("BAJAJFINSV","Financials"),("LICHSGFIN","Financials"),
            ("MUTHOOTFIN","Financials"),("CHOLAFIN","Financials"),
            ("TATACONSUM","Consumer Staples"),("BRITANNIA","Consumer Staples"),
            ("MARICO","Consumer Staples"),("DABUR","Consumer Staples"),
            ("COLPAL","Consumer Staples"),("GODREJCP","Consumer Staples"),
            ("TATAMOTORS","Consumer Discretionary"),("M&M","Consumer Discretionary"),
            ("HEROMOTOCO","Consumer Discretionary"),("BAJAJ-AUTO","Consumer Discretionary"),
            ("EICHERMOT","Consumer Discretionary"),("TVSMOTOR","Consumer Discretionary"),
            ("VOLTAS","Consumer Discretionary"),("HAVELLS","Consumer Discretionary"),
            ("DMART","Consumer Discretionary"),("TRENT","Consumer Discretionary"),
            ("VEDL","Materials"),("NMDC","Materials"),("SAIL","Materials"),
            ("AMBUJACEM","Materials"),("ACC","Materials"),("SHREECEM","Materials"),
            ("ZOMATO","Consumer Discretionary"),("INDIGO","Industrials"),
            ("IRCTC","Industrials"),("ABB","Industrials"),("SIEMENS","Industrials"),
            ("BHEL","Industrials"),("PIDILITIND","Materials"),("BERGEPAINT","Materials"),
            ("TATAPOWER","Utilities"),("ADANIGREEN","Utilities"),("TORNTPOWER","Utilities"),
            ("BIOCON","Health Care"),("AUROPHARMA","Health Care"),("LUPIN","Health Care"),
            ("GLENMARK","Health Care"),("ALKEM","Health Care"),
        ]
        return pd.DataFrame([
            {"Base":b,"Ticker":b+".NS","Sector":s,"NSE Sector":s} for b,s in fallback])

# ─── Quarterly financial helpers ──────────────────────────────────────────────
def _quarterly_revenues(ticker_obj):
    for attr in ("quarterly_income_stmt","quarterly_financials"):
        try:
            df = getattr(ticker_obj, attr)
            if df is None or df.empty: continue
            for label in ["Total Revenue","Revenue","Net Revenue","Operating Revenue"]:
                matches = [r for r in df.index if label.lower() in str(r).lower()]
                if matches:
                    row  = df.loc[matches[0]]
                    cols = sorted(row.index, reverse=True)[:4]
                    vals = [safe_float(row[c]) for c in cols]
                    while len(vals) < 4: vals.append(None)
                    return vals[:4]
        except Exception:
            pass
    return [None, None, None, None]

def _quarterly_eps(ticker_obj):
    for attr in ("quarterly_income_stmt","quarterly_financials"):
        try:
            df = getattr(ticker_obj, attr)
            if df is None or df.empty: continue
            for label in ["Basic EPS","Diluted EPS","EPS"]:
                matches = [r for r in df.index if label.lower() in str(r).lower()]
                if matches:
                    row  = df.loc[matches[0]]
                    cols = sorted(row.index, reverse=True)
                    eps_r = safe_float(row[cols[0]]) if len(cols) > 0 else None
                    eps_o = safe_float(row[cols[3]]) if len(cols) > 3 else None
                    return eps_r, eps_o
        except Exception:
            pass
    return None, None

def _interest_coverage_from_financials(ticker_obj):
    for attr in ("income_stmt","financials"):
        try:
            df = getattr(ticker_obj, attr)
            if df is None or df.empty: continue
            ebit_row = int_row = None
            for label in ["EBIT","Operating Income"]:
                matches = [r for r in df.index if label.lower() in str(r).lower()]
                if matches: ebit_row = df.loc[matches[0]]; break
            for label in ["Interest Expense","Interest Expense Non Operating",
                           "Net Interest Income","Interest And Debt Expense"]:
                matches = [r for r in df.index if label.lower() in str(r).lower()]
                if matches: int_row = df.loc[matches[0]]; break
            if ebit_row is not None and int_row is not None:
                for col in sorted(ebit_row.index, reverse=True)[:2]:
                    ebit = safe_float(ebit_row[col])
                    iexp = safe_float(int_row[col])
                    if ebit is not None and iexp is not None and iexp != 0:
                        ic = ebit / abs(iexp)
                        if ic > 0: return min(ic, 100.0)
        except Exception:
            pass
    return None

# ─── yfinance fundamentals ────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_yf_fundamentals(tickers):
    out = {t: {} for t in tickers}

    def one(t):
        try:
            ticker_obj = yf.Ticker(t)
            info       = ticker_obj.info or {}
            price  = _extract_scalar(info,"currentPrice","regularMarketPrice","previousClose")
            mc     = _extract_scalar(info,"marketCap")
            hi52   = _extract_scalar(info,"fiftyTwoWeekHigh")
            lo52   = _extract_scalar(info,"fiftyTwoWeekLow")
            pe     = _extract_scalar(info,"trailingPE")
            fwd_pe = _extract_scalar(info,"forwardPE")
            peg_yf = _extract_scalar(info,"pegRatio")
            roe    = decimal_to_pct(_extract_scalar(info,"returnOnEquity"))
            roic   = decimal_to_pct(_extract_scalar(info,"returnOnAssets"))
            om     = decimal_to_pct(_extract_scalar(info,"operatingMargins"))
            de_raw = _extract_scalar(info,"debtToEquity")
            ic     = _interest_coverage_from_financials(ticker_obj)
            if ic is None:
                ebitda  = _extract_scalar(info,"ebitda")
                int_exp = _extract_scalar(info,"interestExpense")
                if ebitda and int_exp and int_exp != 0:
                    ic = min(abs(ebitda/int_exp), 100.0)
            eps_r, eps_o = _quarterly_eps(ticker_obj)
            earn_traj = eps_growth = None
            if eps_r is not None and eps_o is not None and abs(eps_o) > 0.001:
                raw = (eps_r-eps_o)/abs(eps_o)
                earn_traj  = max(-1.0, min(1.0, raw/2.0))
                eps_growth = raw*100.0
            if earn_traj is None:
                eps_curr = _extract_scalar(info,"trailingEps")
                eps_fwd  = _extract_scalar(info,"forwardEps")
                if eps_curr and eps_fwd and abs(eps_curr) > 0.001:
                    raw = (eps_fwd-eps_curr)/abs(eps_curr)
                    earn_traj  = max(-1.0, min(1.0, raw/2.0))
                    eps_growth = raw*100.0
            rev4 = _quarterly_revenues(ticker_obj)
            peg  = peg_method = None
            if peg_yf and 0 < peg_yf <= 500:
                peg, peg_method = peg_yf, "yfinance"
            else:
                pe_for_peg = fwd_pe or pe
                if eps_growth and pe_for_peg and float(eps_growth) >= MIN_GROWTH_PCT_FOR_PEG:
                    peg = pe_for_peg/float(eps_growth)
                    peg_method = "Calc"
            if peg and (peg <= 0 or peg > 500): peg = None
            return t, {
                "price":price,"mc":mc,"hi52":hi52,"lo52":lo52,
                "pe":pe if (pe and 0<pe<=10000) else None,
                "fwd_pe":fwd_pe if (fwd_pe and 0<fwd_pe<=10000) else None,
                "peg":peg,"peg_method":peg_method or "N/A",
                "roe":roe,"roic":roic,"op_margin":om,"int_coverage":ic,"debt_eq":de_raw,
                "earn_traj":earn_traj,"eps_growth":eps_growth,"rev4":rev4,
            }
        except Exception:
            return t, {}

    CHUNK=10; SLEEP=0.5; tl=list(tickers)
    chunks=[tl[i:i+CHUNK] for i in range(0,len(tl),CHUNK)]
    prog=st.progress(0); stat=st.empty()
    for ci,chunk in enumerate(chunks):
        stat.text("Fetching fundamentals: {}/{} tickers...".format(min(ci*CHUNK,len(tl)),len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for t,d in ex.map(one,chunk): out[t]=d
        prog.progress((ci+1)/len(chunks))
        if ci<len(chunks)-1: time.sleep(SLEEP)
    prog.empty(); stat.empty()
    return out

# ─── Momentum — per-ticker individual downloads ───────────────────────────────
@st.cache_data(ttl=3600)
def fetch_momentum_batch(tickers):
    """
    Downloads price history individually per ticker.
    Using individual downloads (not bulk) prevents the MultiIndex column
    ambiguity that causes all momentum values to be identical or None.
    """
    tl  = list(tickers)
    out = {t: {} for t in tl}

    def _get_close(df):
        """Safely extract a clean float Series from a single-ticker yf.download result."""
        if df is None or df.empty:
            return pd.Series(dtype=float)
        # Single ticker download returns flat columns — just grab Close
        if isinstance(df.columns, pd.MultiIndex):
            # Shouldn't happen for single-ticker but handle defensively
            try:
                col = df["Close"].iloc[:, 0]
            except Exception:
                return pd.Series(dtype=float)
        else:
            if "Close" not in df.columns:
                return pd.Series(dtype=float)
            col = df["Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return pd.to_numeric(col, errors="coerce").dropna()

    def _process_single(t):
        try:
            # Download daily and monthly separately — always single ticker
            raw_d = yf.download(t, period="7mo", interval="1d",
                                auto_adjust=True, progress=False,
                                group_by="ticker")
            raw_m = yf.download(t, period="7mo", interval="1mo",
                                auto_adjust=True, progress=False,
                                group_by="ticker")

            closes_d = _get_close(raw_d)
            closes_m = _get_close(raw_m)

            if len(closes_m) < 2:
                return t, {}

            px_now = float(closes_m.iloc[-1])

            def ret_mo(n):
                idx = -(n+1)
                if abs(idx) > len(closes_m): return None
                px = float(closes_m.iloc[idx])
                return (px_now/px - 1)*100.0 if px > 0 else None

            r1 = ret_mo(1)
            r3 = ret_mo(3)
            r6 = ret_mo(6)

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

            return t, {
                "ret_1mo": r1, "ret_3mo": r3, "ret_6mo": r6,
                "trailing_vol": trailing_vol, "momentum_score": mom,
            }
        except Exception:
            return t, {}

    # ✅ Serial processing — no threading for momentum
    # Threading + yf.download causes session conflicts that produce
    # identical values across all tickers. Serial is slower but correct.
    CHUNK=5; SLEEP=0.3; chunks=[tl[i:i+CHUNK] for i in range(0,len(tl),CHUNK)]
    prog=st.progress(0); stat=st.empty()
    for ci, chunk in enumerate(chunks):
        stat.text("Fetching momentum: {}/{} tickers...".format(min(ci*CHUNK,len(tl)),len(tl)))
        for t in chunk:
            t_res, d = _process_single(t)
            out[t_res] = d
        prog.progress((ci+1)/len(chunks))
        if ci < len(chunks)-1: time.sleep(SLEEP)
    prog.empty(); stat.empty()
    return out

# ─── Quality ──────────────────────────────────────────────────────────────────
def compute_quality_score(roic, roe, int_coverage, op_margin):
    scores=[]; weights=[]
    prof = roic if roic is not None else roe
    if prof is not None and not pd.isna(prof):
        pf = float(prof)
        scores.append(min(100.0, np.log1p(max(pf,0))/np.log1p(30.0)*100.0) if pf>0 else 0.0)
        weights.append(1.0)
    if int_coverage is not None and not pd.isna(int_coverage):
        scores.append(min(100.0, max(0.0, float(int_coverage)/10.0*100.0)))
        weights.append(1.0)
    if op_margin is not None and not pd.isna(op_margin):
        scores.append(min(100.0, max(0.0, float(op_margin)/40.0*100.0)))
        weights.append(1.0)
    if not scores: return None
    return sum(s*w for s,w in zip(scores,weights))/sum(weights)

def quality_flag(roic, roe, ic, om, de):
    flags=[]
    prof = roic if (roic is not None and not pd.isna(roic)) else roe
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
    scr=scr.copy(); scr["Score"]=pd.NA; scr["Rank"]=pd.NA
    W=FACTOR_WEIGHTS
    for sector in scr["Sector"].dropna().unique():
        elig=scr[(scr["Sector"]==sector)&scr["Eligible"]].copy()
        if elig.empty: continue
        elig["_s_val"]  =percentile_score(elig["P/E"],           ascending=True)
        elig["_s_peg"]  =percentile_score(elig["PEG"],           ascending=True)
        elig["_s_mom"]  =percentile_score(elig["Momentum Score"],ascending=False)
        elig["_s_etraj"]=percentile_score(elig["Earn Traj"],     ascending=False)
        qs=elig["Quality Score"]; q_min,q_max=qs.min(),qs.max()
        elig["_s_quality"]=(
            (qs-q_min)/(q_max-q_min)*100.0
            if pd.notna(q_min) and pd.notna(q_max) and q_max>q_min
            else qs.fillna(0.0))
        elig["_s_quality"]=elig["_s_quality"].fillna(0.0)
        raw=(W["valuation"]*elig["_s_val"]+W["quality"]*elig["_s_quality"]
            +W["peg"]*elig["_s_peg"]+W["earn_traj"]*elig["_s_etraj"]
            +W["momentum"]*elig["_s_mom"])
        pen=elig.apply(lambda r: missing_factor_penalty(
            r,["P/E","PEG","Quality Score","Earn Traj","Momentum Score"]),axis=1)
        elig["Score"]=raw*pen
        elig=elig.sort_values("Score",ascending=False)
        elig["Rank"]=range(1,len(elig)+1)
        scr.loc[elig.index,"Score"]=elig["Score"]
        scr.loc[elig.index,"Rank"]=elig["Rank"]
    return scr

def compute_conviction_scores(scr):
    KEY=["P/E","PEG","Quality Score","Momentum Score","Earn Traj"]
    scr=scr.copy()
    scr["_comp"]=scr.apply(
        lambda r: sum(1 for c in KEY if c in r.index and pd.notna(r[c]))/len(KEY),axis=1)
    med_pe=scr["P/E"].median()
    sec_map=scr.groupby("Sector")["P/E"].median()
    def sec_disc(s):
        if pd.isna(med_pe) or med_pe==0: return 1.0
        sp=sec_map.get(s)
        if pd.isna(sp) or sp==0: return 1.0
        return float(np.clip(med_pe/sp,0.7,1.3))
    scr["_disc"]=scr["Sector"].map(sec_disc)
    raw=scr["Score"]*scr["_comp"]*scr["_disc"]
    cmin,cmax=raw.min(),raw.max()
    scr["Conviction Score"]=(raw-cmin)/(cmax-cmin)*100.0 if cmax>cmin else 50.0
    return scr.drop(columns=["_comp","_disc"])

# ─── Build screener table ─────────────────────────────────────────────────────
def build_screener_table(universe_df, yf_fundamentals, momentum_map):
    rows=[]
    for _,r in universe_df.iterrows():
        t=r["Ticker"]; base=r["Base"]; sec=r["Sector"]
        fd=yf_fundamentals.get(t,{}); mom=momentum_map.get(t,{})
        price=to_num(fd.get("price")); mc=to_num(fd.get("mc"))
        hi52=to_num(fd.get("hi52")); lo52=to_num(fd.get("lo52"))
        pe=to_num(fd.get("pe")); fwd_pe=to_num(fd.get("fwd_pe"))
        peg=to_num(fd.get("peg")); roic=to_num(fd.get("roic"))
        roe=to_num(fd.get("roe")); ic=to_num(fd.get("int_coverage"))
        om=to_num(fd.get("op_margin")); de=to_num(fd.get("debt_eq"))
        earn_traj=to_num(fd.get("earn_traj")); eps_growth=fd.get("eps_growth")
        pos52=None
        if pd.notna(price) and pd.notna(hi52) and pd.notna(lo52) and hi52!=lo52:
            pos52=float(np.clip((price-lo52)/(hi52-lo52)*100.0,0.0,105.0))
        rev4=fd.get("rev4",[None]*4)
        rq1,rq2,rq3,rq4=[to_num(x) for x in rev4]
        growth=revenue_growth_yoy([rq1,rq2,rq3,rq4])
        q_score=compute_quality_score(
            float(roic) if pd.notna(roic) else None,
            float(roe)  if pd.notna(roe)  else None,
            float(ic)   if pd.notna(ic)   else None,
            float(om)   if pd.notna(om)   else None)
        ret_1mo=to_num(mom.get("ret_1mo")); ret_3mo=to_num(mom.get("ret_3mo"))
        ret_6mo=to_num(mom.get("ret_6mo")); mom_score=to_num(mom.get("momentum_score"))
        t_vol=to_num(mom.get("trailing_vol"))
        def to_lcr(v): return float(v)/1e12 if (v is not None and pd.notna(v)) else None
        def to_tcr(v): return float(v)/1e11 if (v is not None and pd.notna(v)) else None
        rows.append({
            "Ticker":base,"YF Ticker":t,"Sector":sec,
            "Price (Rs)":price,
            COL_MC:to_lcr(mc),"Mkt Cap Raw":mc,
            "P/E":pe,"Fwd P/E":fwd_pe,"PEG":peg,
            "Earn Traj":earn_traj,"52W Pos%":to_num(pos52),
            "ROIC% (ROA)":roic,"ROE%":roe,"Int Coverage":ic,
            "Op Margin%":om,"Debt/Eq":de,
            "Quality Score":to_num(q_score) if q_score is not None else None,
            "Momentum Score":mom_score,
            "Ret 1Mo%":ret_1mo,"Ret 3Mo%":ret_3mo,"Ret 6Mo%":ret_6mo,
            "Trailing Vol%":t_vol,"Eligible":True,
            COL_RQ1:to_tcr(rq1),COL_RQ2:to_tcr(rq2),
            COL_RQ3:to_tcr(rq3),COL_RQ4:to_tcr(rq4),
            "Rev Growth% (YoY)":to_num(growth),
        })
    scr=pd.DataFrame(rows)
    if scr.empty: return scr
    total_mc=scr["Mkt Cap Raw"].sum()
    scr["MC% of Index"]=scr["Mkt Cap Raw"]/total_mc*100.0 if total_mc>0 else None
    num_cols=["Price (Rs)",COL_MC,"P/E","Fwd P/E","PEG","52W Pos%",
              "ROIC% (ROA)","ROE%","Int Coverage","Op Margin%","Debt/Eq",
              "Quality Score","Earn Traj","Momentum Score",
              "Ret 1Mo%","Ret 3Mo%","Ret 6Mo%","Trailing Vol%","MC% of Index",
              COL_RQ1,COL_RQ2,COL_RQ3,COL_RQ4,"Rev Growth% (YoY)"]
    for c in num_cols:
        if c in scr.columns: scr[c]=to_num(scr[c])
    scr=compute_rank_by_sector(scr)
    if "Rank" not in scr.columns: scr["Rank"]=pd.NA
    scr=compute_conviction_scores(scr)
    return scr

# ─── Shared screener UI ───────────────────────────────────────────────────────
def render_screener_ui(scr, index_label):
    def _kpi(label, value, sub, color="#ffffff"):
        return (
            "<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;"
            "text-align:center;margin:2px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:4px;'>{}</div>"
            "<div style='color:{};font-size:20px;font-weight:700;'>{}</div>"
            "<div style='color:#666;font-size:10px;margin-top:3px;'>{}</div>"
            "</div>"
        ).format(label, color, value, sub)

    # ✅ FIX: 5 filters in one single row — Sector, Sort by, Min Mkt Cap, Max PE, Max PEG
    f1, f2, f3, f4, f5 = st.columns(5)
    all_sectors = sorted(scr["Sector"].dropna().unique().tolist())
    sector_sel  = f1.selectbox("Sector", ["All Sectors"]+all_sectors, key=index_label+"_sec")
    sort_by     = f2.selectbox("Sort by", [
        "Sector then Rank","Score high to low","Conviction high to low",
        "MC% of Index high to low","Price low to high","Price high to low",
        "Mkt Cap high to low","PE low to high","Fwd PE low to high",
        "PEG low to high","Quality Score high","ROE high to low",
        "Earn Traj high to low","Momentum Score high",
        "52W Pos low to high","Rev Growth high to low",
    ], key=index_label+"_sort")
    mc_min_l = f3.number_input("Min Mkt Cap (LCr)", value=0.0, step=1.0, key=index_label+"_mc")
    pe_max   = f4.number_input("Max PE",            value=9999, step=10,  key=index_label+"_pe")
    peg_max  = f5.number_input("Max PEG",           value=999.0,step=1.0, key=index_label+"_peg")

    # KPI panel
    is_all   = (sector_sel == "All Sectors")
    kpi_label = "{} — All Sectors".format(index_label) if is_all else "{} — {}".format(index_label, sector_sel)
    total_mc = scr["Mkt Cap Raw"].sum()
    sdata    = scr.copy() if is_all else scr[scr["Sector"]==sector_sel]
    sec_mc   = sdata["Mkt Cap Raw"].sum()
    pct      = 100.0 if is_all else (sec_mc/total_mc*100.0 if total_mc>0 else 0.0)
    med_pe   = sdata["P/E"].median()
    med_qual = sdata["Quality Score"].median()
    med_peg  = sdata["PEG"].median()

    st.markdown(
        "<div style='background:#12122a;border:1px solid #2a2a4a;border-radius:12px;"
        "padding:16px 20px;margin-bottom:16px;margin-top:12px;'>"
        "<span style='color:#aaa;font-size:13px;'>Analysis  </span>"
        "<span style='color:#fff;font-size:14px;font-weight:700;'>{}</span>"
        "</div>".format(kpi_label), unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.markdown(_kpi("Sector Mkt Cap",fmt_mc_inr(sec_mc),"Rs Lakh Cr"),unsafe_allow_html=True)
    c2.markdown(_kpi("Index Mkt Cap",fmt_mc_inr(total_mc),"Rs Lakh Cr"),unsafe_allow_html=True)
    c3.markdown(_kpi("Sector Share","{:.1f}%".format(pct),"{} stocks".format(len(sdata))),unsafe_allow_html=True)
    c4.markdown(_kpi("Median P/E",
                     "{:.1f}".format(med_pe) if pd.notna(med_pe) else "N/A",
                     "trailing twelve months","#facc15"),unsafe_allow_html=True)
    c5.markdown(_kpi("Median Quality",
                     "{:.0f}/100".format(med_qual) if pd.notna(med_qual) else "N/A",
                     "ROE+IntCov+Margin","#4ade80"),unsafe_allow_html=True)
    c6.markdown(_kpi("Median PEG",
                     "{:.2f}".format(med_peg) if pd.notna(med_peg) else "N/A",
                     "price/earnings/growth","#a78bfa"),unsafe_allow_html=True)

    if not is_all:
        top3=sdata[sdata["Rank"].notna()].sort_values("Rank").head(3)
        badges="  ".join(
            "<span style='background:#1a2a4a;color:#93c5fd;padding:3px 10px;"
            "border-radius:6px;font-weight:700;font-size:13px;'>{} "
            "<span style='color:#4ade80;font-size:11px;'>#{}</span></span>".format(
                row["Ticker"],int(row["Rank"]))
            for _,row in top3.iterrows())
        st.markdown(
            "<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;margin-bottom:12px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>Top Ranked in Sector</div>"
            "<div>{}</div></div>".format(badges or "<span style='color:#555;'>No ranked stocks</span>"),
            unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

    # Apply filters
    filt=scr.copy()
    if not is_all: filt=filt[filt["Sector"]==sector_sel]
    filt=filt[(filt[COL_MC].isna())|(filt[COL_MC]>=mc_min_l)]
    filt=filt[(filt["P/E"].isna()) |(filt["P/E"]<=pe_max)]
    filt=filt[(filt["PEG"].isna()) |(filt["PEG"]<=peg_max)]

    sort_map={
        "Sector then Rank":         (["Sector","Rank"],[True,True]),
        "Score high to low":        (["Score"],[False]),
        "Conviction high to low":   (["Conviction Score"],[False]),
        "MC% of Index high to low": (["MC% of Index"],[False]),
        "Price low to high":        (["Price (Rs)"],[True]),
        "Price high to low":        (["Price (Rs)"],[False]),
        "Mkt Cap high to low":      ([COL_MC],[False]),
        "PE low to high":           (["P/E"],[True]),
        "Fwd PE low to high":       (["Fwd P/E"],[True]),
        "PEG low to high":          (["PEG"],[True]),
        "Quality Score high":       (["Quality Score"],[False]),
        "ROE high to low":          (["ROE%"],[False]),
        "Earn Traj high to low":    (["Earn Traj"],[False]),
        "Momentum Score high":      (["Momentum Score"],[False]),
        "52W Pos low to high":      (["52W Pos%"],[True]),
        "Rev Growth high to low":   (["Rev Growth% (YoY)"],[False]),
    }
    sc,sa=sort_map.get(sort_by,(["Sector","Rank"],[True,True]))
    filt=filt.sort_values(sc,ascending=sa,na_position="last")

    st.caption("Showing {} of {} stocks · {} · Sort: {}".format(
        len(filt),len(scr),kpi_label,sort_by))

    disp=filt.copy()
    round_cols=["P/E","Fwd P/E","PEG","Earn Traj","52W Pos%",
                "ROIC% (ROA)","ROE%","Int Coverage","Op Margin%","Debt/Eq",
                "Quality Score","Momentum Score","Ret 1Mo%","Ret 3Mo%","Ret 6Mo%",
                "Trailing Vol%","Score","Conviction Score",
                "Rev Growth% (YoY)","MC% of Index","Price (Rs)",
                COL_MC,COL_RQ1,COL_RQ2,COL_RQ3,COL_RQ4]
    for c in round_cols:
        if c in disp.columns: disp[c]=disp[c].round(2)
    disp["Quality Flag"]=disp.apply(
        lambda r: quality_flag(r.get("ROIC% (ROA)"),r.get("ROE%"),
                               r.get("Int Coverage"),r.get("Op Margin%"),r.get("Debt/Eq")),axis=1)
    disp["Rank"]=disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

    COLS=["Ticker","Sector","Price (Rs)","52W Pos%",COL_MC,"MC% of Index",
          "P/E","Fwd P/E","PEG","Earn Traj",
          "ROIC% (ROA)","ROE%","Int Coverage","Op Margin%","Debt/Eq",
          "Quality Score","Quality Flag",
          "Momentum Score","Ret 1Mo%","Ret 3Mo%","Ret 6Mo%","Trailing Vol%",
          "Score","Conviction Score","Rank",
          COL_RQ1,COL_RQ2,COL_RQ3,COL_RQ4,"Rev Growth% (YoY)"]
    disp_final=disp[[c for c in COLS if c in disp.columns]].copy()
    st.dataframe(disp_final,use_container_width=True,height=620)
    st.download_button(
        label="Download CSV",
        data=disp_final.to_csv(index=False).encode("utf-8"),
        file_name="{}_{}.csv".format(index_label.replace(" ","_"),datetime.now().strftime("%Y%m%d_%H%M")),
        mime="text/csv",key=index_label+"_csv")

# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Indian Stock Screener",layout="wide",
                   page_icon="IN",initial_sidebar_state="collapsed")
st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}"
    "</style>",unsafe_allow_html=True)

st.markdown("## Indian Stock Screener")
st.caption("yfinance · Wikipedia universe · 5-factor scoring · INR")

pg1,pg2,pg3,pg4=st.tabs(["Nifty 50","Nifty 500","Stock Comparison","Reference Sheet"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — NIFTY 50
# ══════════════════════════════════════════════════════════════════════════════
with pg1:
    col_r,col_t=st.columns([1,6])
    with col_r:
        if st.button("Refresh",key="n50_refresh"):
            st.cache_data.clear(); st.rerun()
    with col_t:
        st.caption("Last loaded: {} · 1hr cache".format(datetime.now().strftime("%I:%M %p")))

    with st.spinner("Loading Nifty 50 universe..."):
        u50=get_nifty50_universe()
    tickers50=tuple(u50["Ticker"].tolist())

    with st.spinner("Fetching fundamentals ({} stocks)...".format(len(tickers50))):
        fd50=fetch_yf_fundamentals(tickers50)
    with st.spinner("Fetching momentum ({} stocks)...".format(len(tickers50))):
        mom50=fetch_momentum_batch(tickers50)

    total_t=len(tickers50)
    has_price=sum(1 for t in tickers50 if fd50.get(t,{}).get("price") is not None)
    has_mom  =sum(1 for t in tickers50 if mom50.get(t,{}).get("momentum_score") is not None)
    col="info" if has_price>=total_t*0.7 else "warning"
    getattr(st,col)(
        "Coverage — Price: {}/{} ({:.0f}%) · Momentum: {}/{} ({:.0f}%)".format(
            has_price,total_t,has_price/total_t*100,
            has_mom,total_t,has_mom/total_t*100))

    with st.spinner("Building table..."):
        scr50=build_screener_table(u50,fd50,mom50)

    if scr50.empty:
        st.error("No data returned.")
    else:
        render_screener_ui(scr50,"Nifty 50")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — NIFTY 500
# ══════════════════════════════════════════════════════════════════════════════
with pg2:
    st.markdown("### Nifty 500 Screener")
    st.info("Loading 500 stocks takes ~8–12 min on first load. Cached 6 hours after that.")

    col_r2,col_t2=st.columns([1,6])
    with col_r2:
        load_500=st.button("Load Nifty 500",key="n500_load")
        if st.button("Refresh Cache",key="n500_refresh"):
            st.cache_data.clear(); st.rerun()
    with col_t2:
        st.caption("Last loaded: {} · 6hr cache".format(datetime.now().strftime("%I:%M %p")))

    if load_500 or st.session_state.get("n500_loaded",False):
        st.session_state["n500_loaded"]=True
        with st.spinner("Loading Nifty 500 universe..."):
            u500=get_nifty500_universe()
        st.success("Universe: {} stocks".format(len(u500)))
        tickers500=tuple(u500["Ticker"].tolist())

        with st.spinner("Fetching fundamentals for {} stocks...".format(len(tickers500))):
            fd500=fetch_yf_fundamentals(tickers500)
        with st.spinner("Fetching momentum for {} stocks...".format(len(tickers500))):
            mom500=fetch_momentum_batch(tickers500)

        total_t=len(tickers500)
        has_price=sum(1 for t in tickers500 if fd500.get(t,{}).get("price") is not None)
        has_mom  =sum(1 for t in tickers500 if mom500.get(t,{}).get("momentum_score") is not None)
        col="info" if has_price>=total_t*0.6 else "warning"
        getattr(st,col)(
            "Coverage — Price: {}/{} ({:.0f}%) · Momentum: {}/{} ({:.0f}%)".format(
                has_price,total_t,has_price/total_t*100,
                has_mom,total_t,has_mom/total_t*100))

        with st.spinner("Building screener table..."):
            scr500=build_screener_table(u500,fd500,mom500)

        if scr500.empty:
            st.error("No data returned.")
        else:
            render_screener_ui(scr500,"Nifty 500")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STOCK COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with pg3:
    st.markdown("### Stock Comparison")
    st.caption("Compare 2–5 Nifty stocks side by side. Green = best value, Red = worst value per row. Price (Rs) is neutral — no colouring.")

    all_tickers_for_comp=[]
    if "scr50" in dir() and not scr50.empty:
        all_tickers_for_comp=sorted(scr50["Ticker"].dropna().tolist())

    col_pick,col_hint=st.columns([3,2])
    with col_pick:
        selected=st.multiselect(
            "Select 2–5 stocks to compare",
            options=all_tickers_for_comp,
            default=all_tickers_for_comp[:3] if len(all_tickers_for_comp)>=3 else all_tickers_for_comp,
            max_selections=5,key="comp_select")
    with col_hint:
        manual=st.text_input("Or type NSE symbols (comma-separated)",
                             placeholder="e.g. RELIANCE, TCS, INFY",key="comp_manual")
        if manual.strip():
            manual_list=[x.strip().upper() for x in manual.split(",") if x.strip()]
            selected=list(dict.fromkeys(selected+manual_list))[:5]

    if len(selected)<2:
        st.warning("Select at least 2 stocks to compare.")
    else:
        comp_tickers=tuple((s if s.endswith(".NS") else s+".NS") for s in selected)

        with st.spinner("Fetching data for {} stocks...".format(len(comp_tickers))):
            comp_fd =fetch_yf_fundamentals(comp_tickers)
            comp_mom=fetch_momentum_batch(comp_tickers)

        # ── Metrics definition ────────────────────────────────────────────────
        # Each entry: (display_name, extractor_fn, colour_direction)
        # colour_direction: "higher"=green when highest, "lower"=green when lowest, None=no colour
        COMP_METRICS=[
            ("Price (Rs)",     lambda fd,m: fd.get("price"),           None),        # ✅ no colouring
            ("Mkt Cap (LCr)",  lambda fd,m: (fd.get("mc")/1e12) if fd.get("mc") else None, "higher"),
            ("P/E",            lambda fd,m: fd.get("pe"),               "lower"),
            ("Fwd P/E",        lambda fd,m: fd.get("fwd_pe"),           "lower"),
            ("PEG",            lambda fd,m: fd.get("peg"),              "lower"),
            ("Earn Traj",      lambda fd,m: fd.get("earn_traj"),        "higher"),
            ("ROE%",           lambda fd,m: fd.get("roe"),              "higher"),
            ("ROIC% (ROA)",    lambda fd,m: fd.get("roic"),             "higher"),
            ("Op Margin%",     lambda fd,m: fd.get("op_margin"),        "higher"),
            ("Int Coverage",   lambda fd,m: fd.get("int_coverage"),     "higher"),
            ("Debt/Eq",        lambda fd,m: fd.get("debt_eq"),          "lower"),
            ("52W Pos%",       lambda fd,m: (
                float(np.clip((fd.get("price")-fd.get("lo52"))/(fd.get("hi52")-fd.get("lo52"))*100,0,105))
                if all(fd.get(k) for k in ["price","hi52","lo52"]) and fd.get("hi52")!=fd.get("lo52")
                else None), "higher"),
            ("Ret 1Mo%",       lambda fd,m: m.get("ret_1mo"),           "higher"),
            ("Ret 3Mo%",       lambda fd,m: m.get("ret_3mo"),           "higher"),
            ("Ret 6Mo%",       lambda fd,m: m.get("ret_6mo"),           "higher"),
            ("Momentum Score", lambda fd,m: m.get("momentum_score"),    "higher"),
            ("Trailing Vol%",  lambda fd,m: m.get("trailing_vol"),      "lower"),
        ]

        display_names=[t.replace(".NS","") for t in comp_tickers]

        # Build raw values dict
        raw_vals={m[0]: {} for m in COMP_METRICS}
        for t in comp_tickers:
            fd =comp_fd.get(t,{})
            mom=comp_mom.get(t,{})
            name=t.replace(".NS","")
            for metric_name,extractor,_ in COMP_METRICS:
                v=extractor(fd,mom)
                raw_vals[metric_name][name]=(
                    round(float(v),2) if v is not None and not (isinstance(v,float) and np.isnan(v))
                    else None)

        # Build styled HTML table
        header_cells="".join(
            "<th style='padding:10px 14px;background:#1a1a2e;color:#93c5fd;"
            "font-weight:700;text-align:right;border-bottom:2px solid #2a2a4a;'>{}</th>".format(n)
            for n in display_names)
        header_row="<tr><th style='padding:10px 14px;background:#1a1a2e;color:#aaa;"
        "text-align:left;border-bottom:2px solid #2a2a4a;'>Metric</th>{}</tr>".format(header_cells)

        body_rows=[]
        for metric_name,_,direction in COMP_METRICS:
            vals=raw_vals[metric_name]
            nums={k:v for k,v in vals.items() if v is not None}

            # Determine best/worst for colouring
            best_key=worst_key=None
            if direction and len(nums)>=2:
                if direction=="higher":
                    best_key=max(nums,key=nums.__getitem__)
                    worst_key=min(nums,key=nums.__getitem__)
                elif direction=="lower":
                    best_key=min(nums,key=nums.__getitem__)
                    worst_key=max(nums,key=nums.__getitem__)

            metric_cell=(
                "<td style='padding:9px 14px;color:#aaa;font-size:13px;"
                "border-bottom:1px solid #1e1e2e;white-space:nowrap;'>{}</td>".format(metric_name))

            value_cells=[]
            for name in display_names:
                v=vals.get(name)
                disp_v="{:.2f}".format(v) if v is not None else "—"

                # ✅ FIX: Price (Rs) has direction=None so never gets coloured
                if direction is None or best_key is None:
                    style="color:#fff;"
                elif name==best_key:
                    style="color:#4ade80;font-weight:700;"    # green = best
                elif name==worst_key:
                    style="color:#f87171;font-weight:700;"    # red = worst
                else:
                    style="color:#fff;"

                value_cells.append(
                    "<td style='padding:9px 14px;text-align:right;font-size:13px;"
                    "border-bottom:1px solid #1e1e2e;{}'>{}</td>".format(style,disp_v))

            body_rows.append("<tr>{}{}</tr>".format(metric_cell,"".join(value_cells)))

        table_html=(
            "<div style='overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;font-family:Arial,sans-serif;"
            "background:#0d0d1a;border-radius:10px;overflow:hidden;'>"
            "<thead>{}</thead>"
            "<tbody>{}</tbody>"
            "</table></div>"
        ).format(header_row,"".join(body_rows))

        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown(
            "<div style='margin-top:8px;color:#555;font-size:11px;'>"
            "Green = best value in row &nbsp;|&nbsp; Red = worst value in row &nbsp;|&nbsp; "
            "Price (Rs) has no colouring (absolute values, not comparable)"
            "</div>", unsafe_allow_html=True)

        # ── Radar chart ───────────────────────────────────────────────────────
        st.markdown("#### Radar: Normalised Metric Comparison")
        st.caption("Each metric normalised 0–100 within selected stocks. Higher = better on all axes.")

        radar_metrics=["P/E","PEG","ROE%","ROIC% (ROA)","Op Margin%",
                       "Int Coverage","Earn Traj","Momentum Score"]
        lower_better_set={"P/E","PEG","Debt/Eq"}

        radar_data={}
        for t in comp_tickers:
            fd =comp_fd.get(t,{})
            mom=comp_mom.get(t,{})
            radar_data[t.replace(".NS","")]={
                "P/E":fd.get("pe"),"PEG":fd.get("peg"),
                "ROE%":fd.get("roe"),"ROIC% (ROA)":fd.get("roic"),
                "Op Margin%":fd.get("op_margin"),"Int Coverage":fd.get("int_coverage"),
                "Earn Traj":fd.get("earn_traj"),"Momentum Score":mom.get("momentum_score"),
            }

        radar_df=pd.DataFrame(radar_data).T
        norm_df=radar_df.copy()
        for col in radar_df.columns:
            series=pd.to_numeric(radar_df[col],errors="coerce")
            col_min,col_max=series.min(),series.max()
            if pd.isna(col_min) or col_min==col_max:
                norm_df[col]=50.0; continue
            if col in lower_better_set:
                norm_df[col]=(col_max-series)/(col_max-col_min)*100.0
            else:
                norm_df[col]=(series-col_min)/(col_max-col_min)*100.0

        N=len(radar_metrics)
        CX=300; CY=260; R=200
        COLORS=["#93c5fd","#4ade80","#fbbf24","#f87171","#a78bfa"]

        def polar(angle_idx,value,n,cx,cy,r):
            angle=math.pi/2-(2*math.pi*angle_idx/n)
            frac=max(0.0,min(1.0,value/100.0))
            return cx+r*frac*math.cos(angle), cy-r*frac*math.sin(angle)

        svg=[]
        for pct in [0.25,0.5,0.75,1.0]:
            pts=[]
            for i in range(N):
                angle=math.pi/2-(2*math.pi*i/N)
                pts.append("{:.1f},{:.1f}".format(CX+R*pct*math.cos(angle),CY-R*pct*math.sin(angle)))
            svg.append("<polygon points='{}' fill='none' stroke='#2a2a4a' stroke-width='1'/>".format(" ".join(pts)))
        for i,m in enumerate(radar_metrics):
            angle=math.pi/2-(2*math.pi*i/N)
            x2,y2=CX+R*math.cos(angle),CY-R*math.sin(angle)
            lx,ly=CX+(R+28)*math.cos(angle),CY-(R+28)*math.sin(angle)
            svg.append("<line x1='{:.1f}' y1='{:.1f}' x2='{:.1f}' y2='{:.1f}' stroke='#2a2a4a' stroke-width='1'/>".format(CX,CY,x2,y2))
            anchor="middle" if abs(math.cos(angle))<0.3 else ("start" if math.cos(angle)>0 else "end")
            svg.append("<text x='{:.1f}' y='{:.1f}' fill='#aaa' font-size='11' text-anchor='{}' dominant-baseline='middle' font-family='Arial'>{}</text>".format(lx,ly,anchor,m))
        for si,name in enumerate(norm_df.index):
            color=COLORS[si%len(COLORS)]
            pts=[]
            for i,m in enumerate(radar_metrics):
                v=norm_df.loc[name,m]; v=0.0 if pd.isna(v) else float(v)
                x,y=polar(i,v,N,CX,CY,R)
                pts.append("{:.1f},{:.1f}".format(x,y))
            svg.append("<polygon points='{}' fill='{}' fill-opacity='0.15' stroke='{}' stroke-width='2'/>".format(" ".join(pts),color,color))
            for i,m in enumerate(radar_metrics):
                v=norm_df.loc[name,m]; v=0.0 if pd.isna(v) else float(v)
                x,y=polar(i,v,N,CX,CY,R)
                svg.append("<circle cx='{:.1f}' cy='{:.1f}' r='4' fill='{}' stroke='#000' stroke-width='1'/>".format(x,y,color))
        leg_y=540
        for si,name in enumerate(norm_df.index):
            color=COLORS[si%len(COLORS)]; lx=80+si*110
            svg.append("<rect x='{}' y='{}' width='12' height='12' fill='{}' rx='2'/>".format(lx,leg_y,color))
            svg.append("<text x='{}' y='{}' fill='#fff' font-size='12' font-family='Arial' dominant-baseline='middle'>{}</text>".format(lx+16,leg_y+6,name))

        st.markdown(
            "<div style='display:flex;justify-content:center;'>"
            "<svg width='600' height='580' style='background:#0d0d1a;border-radius:12px;'>"
            +"".join(svg)+
            "</svg></div>",unsafe_allow_html=True)

        # Download
        comp_export=pd.DataFrame({m[0]:{t.replace(".NS",""):raw_vals[m[0]].get(t.replace(".NS",""))
                                        for t in comp_tickers}
                                   for m in COMP_METRICS}).T
        st.download_button(
            label="Download Comparison CSV",
            data=comp_export.to_csv().encode("utf-8"),
            file_name="comparison_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
            mime="text/csv",key="comp_csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REFERENCE SHEET
# ══════════════════════════════════════════════════════════════════════════════
with pg4:
    st.markdown("## Reference Sheet — How Every Metric Works")
    st.caption("Formulas, numeric examples, scoring logic, and benchmarks for every column in the screener.")

    tab_val,tab_qual,tab_peg,tab_etraj,tab_mom,tab_rank,tab_disp=st.tabs([
        "Valuation","Quality","PEG","Earn Traj","Momentum","Scoring & Rank","Display Metrics"])

    with tab_val:
        st.markdown("""
### P/E — Price to Earnings Ratio (Trailing)
**What it is:** How many rupees you pay per rupee of actual profit the company earned over the last 12 months.

**Formula:** `Current Stock Price / Trailing 12-Month EPS`

**Numeric Example (Indian context):**
- Reliance Industries price = Rs.1,297. Reliance earned Rs.65 per share TTM.
- P/E = 1297 / 65 = **19.9**
- Meaning: You pay Rs.19.90 for every Rs.1 of Reliance's annual profit.

**Sector median P/E benchmarks (typical India ranges):**

| Sector | Typical Median P/E |
|--------|-------------------|
| Information Technology | 24–32 |
| Consumer Staples (FMCG) | 40–55 |
| Financials / Banking | 12–18 |
| Energy / Oil & Gas | 8–14 |
| Industrials | 20–30 |
| Health Care / Pharma | 22–35 |
| Materials / Metals | 8–15 |
| Utilities | 12–18 |
| Consumer Discretionary | 25–45 |

**Used in scoring?** Yes — primary Valuation factor (25% weight). Lower P/E = better percentile within sector.

---

### Fwd P/E — Forward Price to Earnings
**What it is:** Same as P/E but uses analysts' consensus EPS estimate for the next 12 months.

**Formula:** `Current Stock Price / Next 12-Month Estimated EPS`

**Numeric Example:**
- Infosys price = Rs.1,450, trailing EPS = Rs.59, forward EPS estimate = Rs.68
- Trailing P/E = 24.6, **Fwd P/E = 1450 / 68 = 21.3**
- Fwd P/E < Trailing P/E → earnings expected to grow ~15% → positive signal

**Used in scoring?** Yes — preferred over trailing P/E for Valuation factor.

---

### MC% of Index
**What it is:** Stock's share of total index market capitalisation.

**Formula:** `Stock Market Cap / Sum of All Index Stock Market Caps x 100`

**Used in scoring?** No. Display and filter only.

---

### 52W Pos%
**Formula:** `(Current Price - 52W Low) / (52W High - 52W Low) x 100`

0% = at 52-week low. 100% = at 52-week high. **Display only.**
""")

    with tab_qual:
        st.markdown("""
### Quality Score (0–100)
**Formula:** `(ROIC sub-score + Int Coverage sub-score + Op Margin sub-score) / 3`

**Used in scoring?** Yes — 25% weight.

---

### ROIC% (ROA proxy)
**What it is:** Return on Assets used as a proxy for Return on Invested Capital.

**Formula:** `Net Income / Total Assets x 100`

**Benchmarks:**

| ROA (ROIC proxy) | Quality sub-score |
|-----------------|------------------|
| 25%+ | ~90–100 |
| 15% | ~65 |
| 8% | ~49 (minimum threshold) |
| Below 0% | 0 + flag |

---

### Int Coverage
**Formula:** `EBIT / |Interest Expense|` from annual income statement.

**Benchmarks:** 10x+ = full marks (100), 3x = flagged (30), below 1x = distress (0).

**Note for banks/NBFCs:** Not meaningful — their business model is borrowing and lending.

---

### Op Margin%
**Formula:** `Operating Income / Revenue x 100`

**Benchmarks:** 40%+ = elite (100), 20% = good (50), below 5% = flagged (0).

---

### Quality Flag
- `ROIC<8%` — capital return below minimum
- `IntCov<3x` — debt servicing risk
- `Margin<5%` — thin profitability
- `Pass` — all checks cleared
- `D/E: x.x` — informational, always shown
""")

    with tab_peg:
        st.markdown("""
### PEG — Price/Earnings-to-Growth Ratio
**Formula:** `P/E / Annual EPS Growth Rate (%)`

**Interpreting PEG:**
- Below 1.0: Potentially undervalued
- 1.0–2.0: Fairly valued
- Above 2.0: Expensive relative to growth
- Above 3.0: Very hard to justify

**Growth guard:** Only computed when EPS growth >= 5%.

| Stock | P/E | EPS Growth | PEG | Verdict |
|-------|-----|-----------|-----|---------|
| Bajaj Finance | 28 | 25% | 1.12 | Fairly valued |
| HDFC Bank | 18 | 18% | 1.00 | Perfectly priced |
| ITC | 28 | 8% | 3.50 | Expensive for growth |
| Zomato | 120 | 80% | 1.50 | Reasonable for hypergrowth |

**Used in scoring?** Yes — 20% weight. Lower = better.
""")

    with tab_etraj:
        st.markdown("""
### Earn Traj — Earnings Trajectory
**Formula:** `(Forward EPS - Trailing EPS) / |Trailing EPS|` clipped to [-1.0, +1.0]

Raw ratio divided by 2 before clipping to preserve nuance for moderate changes.

**Interpreting:**
- +0.5 to +1.0: Strong expected earnings growth
- +0.1 to +0.3: Moderate growth
- Near 0: Flat earnings
- -0.1 to -0.3: Earnings under pressure
- -0.5 to -1.0: Significant deterioration

**Practical filters:**
- Min Earn Traj = 0.10 → only stocks forecasting 10%+ EPS growth
- Min Earn Traj = 0.25 → high-conviction growth stories only

**Used in scoring?** Yes — 15% weight. Higher = better.
""")

    with tab_mom:
        st.markdown("""
### Momentum Score
**Formula:** `(6-month return - 1-month return) / Trailing 90-day Annualised Volatility`

**Why subtract 1-month (skip-month)?** The most recent month has a documented short-term reversal. Removing it isolates the durable 2–6 month trend.

**Example:**
- HDFC Bank: 6mo return = +18%, 1mo = +3%, vol = 16% → Score = 15/16 = **0.94**
- Adani Ent: 6mo return = +35%, 1mo = +8%, vol = 48% → Score = 27/48 = **0.56**

HDFC Bank scores higher despite lower raw return — its move is more signal-rich per unit of risk.

**Used in scoring?** Yes — 15% weight. Higher = better.

---

### Ret 1Mo%, Ret 3Mo%, Ret 6Mo%
Raw percentage price returns from monthly closes. **Display only.**

---

### Trailing Vol%
`Daily Return Std Dev x sqrt(252) x 100` over last 90 days.

**Display only** — denominator in Momentum Score.
""")

    with tab_rank:
        st.markdown("""
### Score (0–100)
All percentiles are computed **within each sector independently**. Score 70 in IT and Score 70 in Utilities both mean top-30% of their sector — not directly comparable across sectors.

---

### Missing Factor Penalty

| Missing factors | Multiplier |
|----------------|-----------|
| 0–1 | x1.00 |
| 2 | x0.85 |
| 3+ | x0.70 |

---

### Rank
Ordinal position within sector by Score. Rank 1 = best in that sector.

---

### Conviction Score (0–100)
`Score x data_completeness x sector_discount → normalised 0–100`

- **Completeness:** stocks with fewer data points get reduced confidence
- **Sector discount:** cheap sectors (Energy, Metals) get a boost up to +30%; expensive sectors (FMCG) get a gentle penalty up to -30%
""")

    with tab_disp:
        st.markdown("""
### ROE% — Return on Equity (Display Only)
`Net Income / Shareholders Equity x 100`

Display only — distorted by leverage and buybacks. ROIC/ROA is more reliable for scoring.

---

### Debt/Eq
`Total Debt / Total Shareholders Equity`

Display only — context for Quality Flag. Interest Coverage is what's scored.

---

### Rev Q1–Q4 (Rs 1000 Cr)
Last 4 quarterly revenues, newest first. 1 unit = Rs 1,000 Cr.

Accelerating pattern (positive): Q4=14 → Q3=16 → Q2=18 → Q1=20

---

### Rev Growth% (YoY)
`(Newest quarter / Same quarter 1 year ago - 1) x 100`

Removes seasonal effects by comparing same quarter year-over-year. **Display only.**

---

### Data Coverage (typical for .NS tickers)

| Metric | Coverage |
|--------|---------|
| Price, MC, 52W | ~99% |
| Trailing P/E | ~90% |
| Forward P/E | ~75% |
| PEG | ~70% |
| ROE, Op Margin | ~85% |
| Int Coverage | ~65% |
| Earn Traj | ~80% |
| Momentum | ~95% |
| Quarterly Revenue | ~70% |
""")

    st.markdown("---")
    st.markdown(
        "**Data source:** Yahoo Finance (yfinance) — all fundamentals and price history. "
        "Universe from Wikipedia NIFTY_50 / NIFTY_500. "
        "Nothing here is financial advice.")
