# screener_app_nifty50.py  v2
# ─────────────────────────────────────────────────────────────────────────────
# Nifty 50 Fundamental Screener — ported from S&P 500 Screener v8
#
# v2 fixes:
#   1. Removed duplicate get_nifty50_universe() — Wikipedia scraper now active
#   2. Fixed footnote regex: $.*?$ → $.*?$
#   3. Added BeautifulSoup import guard
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

# ── Universe (Wikipedia scraper — single definition) ──────────────────────────
@st.cache_data(ttl=86400)
def get_nifty50_universe():
    """
    Scrapes Nifty 50 constituents dynamically from Wikipedia.
    URL: https://en.wikipedia.org/wiki/NIFTY_50
    Falls back to minimal hardcoded list if scrape fails.
    """
    url = "https://en.wikipedia.org/wiki/NIFTY_50"

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

    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # ── Find the constituents table (4 strategies) ────────────────────
        table = soup.find("table", {"id": "constituents"})

        if table is None:
            for tbl in soup.find_all("table", {"class": "wikitable"}):
                headers = [th.get_text(strip=True).lower()
                           for th in tbl.find_all("th")]
                if any("symbol" in h or "ticker" in h for h in headers):
                    table = tbl
                    break

        if table is None:
            table = soup.find("table", {"class": "wikitable sortable"})

        if table is None:
            for tbl in soup.find_all("table", {"class": "wikitable"}):
                if len(tbl.find_all("tr")) >= 30:
                    table = tbl
                    break

        if table is None:
            raise RuntimeError("No suitable table found on Wikipedia NIFTY_50 page")

        # ── Detect column positions from headers ──────────────────────────
        header_row = table.find("tr")
        headers    = []
        if header_row:
            headers = [th.get_text(strip=True).lower()
                       for th in header_row.find_all(["th", "td"])]

        ticker_col = None
        sector_col = None

        for i, h in enumerate(headers):
            if any(kw in h for kw in ["symbol", "ticker", "nse"]):
                ticker_col = i
            if any(kw in h for kw in ["sector", "industry", "gics"]):
                sector_col = i

        # Fallback positions if header detection fails
        if ticker_col is None:
            ticker_col = 2
        if sector_col is None:
            sector_col = 1

        # ── Extract rows ──────────────────────────────────────────────────
        data = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) <= max(ticker_col, sector_col):
                continue

            # ✅ Fixed regex — strip Wikipedia footnotes like [a], [1]
            raw_ticker = cols[ticker_col].get_text(strip=True)
            raw_ticker = re.sub(r"$.*?$", "", raw_ticker).strip()
            raw_ticker = re.sub(r"[^A-Za-z0-9&\-]", "", raw_ticker).upper()

            if not raw_ticker or len(raw_ticker) < 2:
                continue

            raw_sector = cols[sector_col].get_text(strip=True)
            raw_sector = re.sub(r"$.*?$", "", raw_sector).strip()

            # Map NSE sector → GICS
            gics_sector = SECTOR_MAP.get(raw_sector)
            if gics_sector is None:
                for nse_name, gics_name in SECTOR_MAP.items():
                    if (nse_name.lower() in raw_sector.lower()
                            or raw_sector.lower() in nse_name.lower()):
                        gics_sector = gics_name
                        break
            if gics_sector is None:
                gics_sector = raw_sector  # keep original if no mapping

            data.append({
                "Ticker":     raw_ticker + ".NS",
                "NSE Symbol": raw_ticker,
                "Sector":     gics_sector,
                "NSE Sector": raw_sector,
            })

        if len(data) < 30:
            raise RuntimeError(
                "Only {} rows parsed — Wikipedia table may have changed".format(
                    len(data)))

        df = pd.DataFrame(data).drop_duplicates(subset=["Ticker"])
        st.success("✅ Loaded {} stocks across {} sectors from Wikipedia".format(
            len(df), df["Sector"].nunique()))
        return df

    except Exception as e:
        st.warning("⚠️ Wikipedia scrape failed: {}. Using fallback list.".format(str(e)))
        fallback = [
            {"Ticker": "RELIANCE.NS",   "NSE Symbol": "RELIANCE",   "Sector": "Energy",                 "NSE Sector": "Oil Gas & Consumable Fuels"},
            {"Ticker": "TCS.NS",        "NSE Symbol": "TCS",        "Sector": "Information Technology", "NSE Sector": "Information Technology"},
            {"Ticker": "HDFCBANK.NS",   "NSE Symbol": "HDFCBANK",   "Sector": "Financials",             "NSE Sector": "Financial Services"},
            {"Ticker": "INFY.NS",       "NSE Symbol": "INFY",       "Sector": "Information Technology", "NSE Sector": "Information Technology"},
            {"Ticker": "ICICIBANK.NS",  "NSE Symbol": "ICICIBANK",  "Sector": "Financials",             "NSE Sector": "Financial Services"},
            {"Ticker": "HINDUNILVR.NS", "NSE Symbol": "HINDUNILVR", "Sector": "Consumer Staples",       "NSE Sector": "FMCG"},
            {"Ticker": "ITC.NS",        "NSE Symbol": "ITC",        "Sector": "Consumer Staples",       "NSE Sector": "FMCG"},
            {"Ticker": "SBIN.NS",       "NSE Symbol": "SBIN",       "Sector": "Financials",             "NSE Sector": "Financial Services"},
            {"Ticker": "BHARTIARTL.NS", "NSE Symbol": "BHARTIARTL", "Sector": "Communication Services", "NSE Sector": "Telecommunication"},
            {"Ticker": "LT.NS",         "NSE Symbol": "LT",         "Sector": "Industrials",            "NSE Sector": "Construction"},
        ]
        return pd.DataFrame(fallback)


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def sf(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

def normalise_pct(val):
    if val is None:
        return None
    v = float(val)
    return v * 100.0 if abs(v) < 5.0 else v

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


# ── Prices batch ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices_batch(tickers):
    tl  = list(tickers)
    res = {t: {"price": None, "hi52": None, "lo52": None, "mc": None} for t in tl}
    try:
        raw = yf.download(
            tl, period="2d", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True
        )
        for t in tl:
            try:
                px = (float(raw["Close"].iloc[-1]) if len(tl) == 1
                      else float(raw[t]["Close"].iloc[-1]))
                res[t]["price"] = px
            except Exception:
                pass
    except Exception:
        pass
    return res


# ── Momentum batch ────────────────────────────────────────────────────────────
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
                    if abs(idx) > len(closes_m):
                        return None
                    px = float(closes_m.iloc[idx])
                    return (px_now / px - 1) * 100.0 if px > 0 else None

                r1 = ret_mo(1)
                r3 = ret_mo(3)
                r6 = ret_mo(6)

                trailing_vol = None
                if len(closes_d) >= 20:
                    daily_rets = closes_d.pct_change().dropna().tail(90)
                    if len(daily_rets) >= 15:
                        trailing_vol = float(daily_rets.std() * np.sqrt(252) * 100.0)

                skip_mom_raw = ((r6 - r1) if (r6 is not None and r1 is not None)
                                else None)
                skip_mom_adj = None
                if skip_mom_raw is not None and trailing_vol and trailing_vol > 0:
                    skip_mom_adj = skip_mom_raw / trailing_vol
                elif skip_mom_raw is not None:
                    skip_mom_adj = skip_mom_raw

                out[t] = {
                    "ret_1mo":        r1,
                    "ret_3mo":        r3,
                    "ret_6mo":        r6,
                    "trailing_vol":   trailing_vol,
                    "momentum_score": skip_mom_adj,
                }
            except Exception:
                pass
    except Exception:
        pass
    return out


# ── Yahoo fundamentals (single ticker) ───────────────────────────────────────
def _fetch_yahoo_fundamentals_one(t):
    result = {
        "pe": None, "pe_src": None,
        "fwd_pe": None,
        "peg": None, "peg_src": None,
        "roe": None,
        "roic": None,
        "op_margin": None,
        "debt_eq": None,
        "eps_growth": None,
        "int_coverage": None,
        "earn_traj": None,
        "mc": None,
        "hi52": None,
        "lo52": None,
    }
    try:
        obj = yf.Ticker(t)

        try:
            fi = obj.fast_info
            if fi is not None:
                mc_fi = sf(getattr(fi, "market_cap", None))
                hi_fi = sf(getattr(fi, "year_high",  None))
                lo_fi = sf(getattr(fi, "year_low",   None))
                if mc_fi: result["mc"]   = mc_fi
                if hi_fi: result["hi52"] = hi_fi
                if lo_fi: result["lo52"] = lo_fi
        except Exception:
            pass

        info = {}
        for attempt in range(2):
            try:
                info = obj.info or {}
                if info.get("trailingPE") or info.get("pegRatio") or info.get("forwardPE"):
                    break
            except Exception:
                pass
            time.sleep(0.5 + random.uniform(0, 0.5))

        px = sf(info.get("currentPrice") or info.get("regularMarketPrice"))

        t_pe  = sf(info.get("trailingPE"))
        t_eps = sf(info.get("trailingEps"))
        if t_pe and 0 < t_pe <= 10_000:
            result["pe"]     = t_pe
            result["pe_src"] = "Yahoo"
        elif t_eps and t_eps > 0 and px and px > 0:
            result["pe"]     = px / t_eps
            result["pe_src"] = "Yahoo(calc)"

        f_pe  = sf(info.get("forwardPE"))
        f_eps = sf(info.get("forwardEps"))
        if f_pe and 0 < f_pe <= 10_000:
            result["fwd_pe"] = f_pe
        elif f_eps and f_eps > 0 and px and px > 0:
            result["fwd_pe"] = px / f_eps

        peg_y = sf(info.get("pegRatio"))
        if peg_y and 0 < peg_y <= 500:
            result["peg"]     = peg_y
            result["peg_src"] = "Yahoo"

        roe_y = sf(info.get("returnOnEquity"))
        if roe_y is not None:
            result["roe"] = roe_y * 100.0

        om_y = sf(info.get("operatingMargins"))
        if om_y is not None:
            result["op_margin"] = om_y * 100.0

        de_y = sf(info.get("debtToEquity"))
        if de_y is not None:
            result["debt_eq"] = de_y / 100.0

        eg_y = sf(info.get("earningsGrowth"))
        if eg_y is not None:
            result["eps_growth"] = eg_y * 100.0

        fwd_eps_val   = sf(info.get("forwardEps"))
        trail_eps_val = sf(info.get("trailingEps"))
        if (fwd_eps_val is not None and trail_eps_val is not None
                and abs(trail_eps_val) > 0.01):
            earn_traj_raw       = (fwd_eps_val - trail_eps_val) / abs(trail_eps_val)
            result["earn_traj"] = max(-1.0, min(1.0, earn_traj_raw))

        if result["mc"] is None:
            mc_y = sf(info.get("marketCap"))
            if mc_y:
                result["mc"] = mc_y

        if result["hi52"] is None:
            h52 = sf(info.get("fiftyTwoWeekHigh"))
            if h52: result["hi52"] = h52
        if result["lo52"] is None:
            l52 = sf(info.get("fiftyTwoWeekLow"))
            if l52: result["lo52"] = l52

        try:
            qfin = obj.quarterly_financials
            if qfin is not None and not qfin.empty:
                ebit_row = None
                for nm in ["EBIT", "Operating Income", "Ebit"]:
                    if nm in qfin.index:
                        ebit_row = nm; break
                int_row = None
                for nm in ["Interest Expense",
                           "Interest Expense Non Operating",
                           "Net Interest Income"]:
                    if nm in qfin.index:
                        int_row = nm; break
                if ebit_row and int_row:
                    ebit_ttm = qfin.loc[ebit_row].dropna().head(4).sum()
                    int_ttm  = abs(qfin.loc[int_row].dropna().head(4).sum())
                    if int_ttm > 0 and ebit_ttm > 0:
                        result["int_coverage"] = min(float(ebit_ttm / int_ttm), 100.0)
        except Exception:
            pass

        try:
            qfin = obj.quarterly_financials
            bs   = obj.quarterly_balance_sheet
            if (qfin is not None and not qfin.empty
                    and bs is not None and not bs.empty):
                op_inc_row = None
                for nm in ["Operating Income", "EBIT", "Ebit"]:
                    if nm in qfin.index:
                        op_inc_row = nm; break
                tax_row = None
                for nm in ["Tax Provision", "Income Tax Expense", "Tax Expense"]:
                    if nm in qfin.index:
                        tax_row = nm; break
                pretax_row = None
                for nm in ["Pretax Income", "Income Before Tax", "EBT"]:
                    if nm in qfin.index:
                        pretax_row = nm; break
                if op_inc_row:
                    op_inc_ttm   = float(qfin.loc[op_inc_row].dropna().head(4).sum())
                    eff_tax_rate = 0.25  # India default
                    if tax_row and pretax_row:
                        tax_ttm    = float(qfin.loc[tax_row].dropna().head(4).sum())
                        pretax_ttm = float(qfin.loc[pretax_row].dropna().head(4).sum())
                        if pretax_ttm > 0 and tax_ttm >= 0:
                            computed_rate = tax_ttm / pretax_ttm
                            if 0 < computed_rate < 0.6:
                                eff_tax_rate = computed_rate
                    nopat      = op_inc_ttm * (1 - eff_tax_rate)
                    equity_val = None
                    for nm in ["Total Stockholders Equity", "Stockholders Equity",
                               "Common Stock Equity",
                               "Total Equity Gross Minority Interest"]:
                        if nm in bs.index:
                            eq_s = bs.loc[nm].dropna()
                            if len(eq_s) > 0:
                                equity_val = float(eq_s.iloc[0]); break
                    debt_val = None
                    for nm in ["Total Debt", "Net Debt", "Long Term Debt",
                               "Long Term Debt And Capital Lease Obligation"]:
                        if nm in bs.index:
                            d_s = bs.loc[nm].dropna()
                            if len(d_s) > 0:
                                debt_val = float(d_s.iloc[0]); break
                    cash_val = None
                    for nm in ["Cash And Cash Equivalents",
                               "Cash Cash Equivalents And Short Term Investments",
                               "Cash Financial", "Cash And Short Term Investments"]:
                        if nm in bs.index:
                            c_s = bs.loc[nm].dropna()
                            if len(c_s) > 0:
                                cash_val = float(c_s.iloc[0]); break
                    if equity_val is not None and debt_val is not None:
                        cash_use         = cash_val if cash_val is not None else 0
                        invested_capital = equity_val + debt_val - cash_use
                        if invested_capital > 0 and nopat != 0:
                            roic_computed = (nopat / invested_capital) * 100.0
                            if -100 < roic_computed < 200:
                                result["roic"] = roic_computed
        except Exception:
            pass

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_fundamentals_all(tickers):
    tl     = list(tickers)
    out    = {}
    CHUNK  = 10
    WKRS   = 5
    SLEEP  = 1.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    progress = st.progress(0)
    status   = st.empty()
    total    = len(chunks)
    for ci, chunk in enumerate(chunks):
        status.text("Yahoo fundamentals {}/{} ({} of {} done)...".format(
            ci+1, total, ci*CHUNK, len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futures = {ex.submit(_fetch_yahoo_fundamentals_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    t, d = fut.result()
                    out[t] = d
                except Exception:
                    t = futures[fut]
                    out[t] = {}
        progress.progress((ci + 1) / total)
        if ci < len(chunks) - 1:
            time.sleep(SLEEP + random.uniform(0, 0.5))
    progress.empty()
    status.empty()
    return out


# ── Revenue ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_last4_revenue_parallel(tickers):
    tl  = list(tickers)
    out = {}

    def one(t):
        try:
            qf = yf.Ticker(t).quarterly_financials
            if qf is not None and "Total Revenue" in qf.index:
                s = qf.loc["Total Revenue"].sort_index().tail(4)
                v = [float(x) for x in s.values]
                if len(v) == 4:
                    return t, v
        except Exception:
            pass
        return t, [None, None, None, None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for t, v in ex.map(one, tl):
            out[t] = v
    return out


# ── Quality Score ─────────────────────────────────────────────────────────────
def compute_quality_score(roic, roe, int_coverage, op_margin):
    scores = []
    profitability = roic if roic is not None else roe
    if profitability is not None and not pd.isna(profitability):
        pf = float(profitability)
        scores.append(
            min(100.0, np.log1p(pf) / np.log1p(30.0) * 100.0) if pf > 0 else 0.0)
    else:
        scores.append(0.0)
    if int_coverage is not None and not pd.isna(int_coverage):
        scores.append(min(100.0, max(0.0, float(int_coverage) / 10.0 * 100.0)))
    else:
        scores.append(0.0)
    if op_margin is not None and not pd.isna(op_margin):
        scores.append(min(100.0, max(0.0, float(op_margin) / 40.0 * 100.0)))
    else:
        scores.append(0.0)
    return sum(scores) / 3.0


# ── Quality flag ──────────────────────────────────────────────────────────────
def quality_flag(roic, roe, ic, om, de):
    flags = []
    profitability = roic if (roic is not None and not pd.isna(roic)) else roe
    if (profitability is not None and not pd.isna(profitability)
            and profitability < QUALITY_THRESHOLDS["roic_min"]):
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
        g    = scr[scr["Sector"] == sector].copy()
        elig = g[g["Eligible"]].copy()
        if elig.empty:
            continue

        pe_input         = elig["Fwd P/E"].fillna(elig["P/E"])
        elig["_s_val"]   = percentile_score(pe_input,               ascending=True)
        elig["_s_peg"]   = percentile_score(elig["PEG"],            ascending=True)
        elig["_s_mom"]   = percentile_score(elig["Momentum Score"], ascending=False)
        elig["_s_etraj"] = percentile_score(elig["Earn Traj"],      ascending=False)

        qs    = elig["Quality Score"]
        q_min = qs.min(); q_max = qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        raw = (W["valuation"] * elig["_s_val"]
               + W["quality"]   * elig["_s_quality"]
               + W["peg"]       * elig["_s_peg"]
               + W["earn_traj"] * elig["_s_etraj"]
               + W["momentum"]  * elig["_s_mom"])

        factor_cols = ["P/E", "PEG", "Quality Score", "Earn Traj", "Momentum Score"]
        penalties   = elig.apply(lambda r: missing_factor_penalty(r, factor_cols), axis=1)
        raw         = raw * penalties

        elig["Score"] = raw
        elig          = elig.sort_values("Score", ascending=False)
        elig["Rank"]  = range(1, len(elig) + 1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]

    return scr


def compute_conviction_scores(scr):
    KEY_FACTORS = ["P/E", "Fwd P/E", "PEG", "Quality Score", "Momentum Score", "Earn Traj"]
    n_factors   = len(KEY_FACTORS)
    scr         = scr.copy()

    def completeness(row):
        return sum(1 for c in KEY_FACTORS
                   if c in row.index and pd.notna(row[c])) / n_factors

    scr["_completeness"] = scr.apply(completeness, axis=1)
    overall_median_pe    = scr["P/E"].median()
    sector_pe_map        = scr.groupby("Sector")["P/E"].median()

    def sector_discount(sector):
        if pd.isna(overall_median_pe) or overall_median_pe == 0:
            return 1.0
        s_pe = sector_pe_map.get(sector)
        if pd.isna(s_pe) or s_pe == 0:
            return 1.0
        return float(np.clip(overall_median_pe / s_pe, 0.7, 1.3))

    scr["_sec_discount"] = scr["Sector"].map(sector_discount)
    raw_conviction       = scr["Score"] * scr["_completeness"] * scr["_sec_discount"]
    c_min, c_max         = raw_conviction.min(), raw_conviction.max()
    scr["Conviction Score"] = ((raw_conviction - c_min) / (c_max - c_min) * 100.0
                               if c_max > c_min else 50.0)
    return scr.drop(columns=["_completeness", "_sec_discount"])


# ── Build table ───────────────────────────────────────────────────────────────
def build_screener_table(universe_df, prices_map, yahoo_map, revenue_map, momentum_map):
    rows = []
    for _, r in universe_df.iterrows():
        t   = r["Ticker"]
        sec = r["Sector"]

        px_info   = prices_map.get(t, {})
        price     = to_num(px_info.get("price"))
        fi        = yahoo_map.get(t, {})
        mc        = to_num(fi.get("mc"))
        pe        = to_num(fi.get("pe"))
        fwd       = to_num(fi.get("fwd_pe"))
        hi        = to_num(fi.get("hi52"))
        lo        = to_num(fi.get("lo52"))
        roic      = to_num(fi.get("roic"))
        roe       = to_num(fi.get("roe"))
        ic        = to_num(fi.get("int_coverage"))
        om        = to_num(fi.get("op_margin"))
        de        = to_num(fi.get("debt_eq"))
        earn_traj = to_num(fi.get("earn_traj"))

        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price - lo) / (hi - lo) * 100.0)

        rev4                = revenue_map.get(t, [None]*4)
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth              = revenue_growth_pct_cagr([rq1, rq2, rq3, rq4])

        peg_direct = to_num(fi.get("peg"))
        peg = None; peg_method = "—"
        if pd.notna(peg_direct):
            peg        = float(peg_direct)
            peg_method = fi.get("peg_src") or "Yahoo"
        else:
            pe_for_peg = fwd if pd.notna(fwd) else pe
            eps_g      = fi.get("eps_growth")
            g_src      = fi.get("growth_src") or ""
            if eps_g is not None:
                eg = float(eps_g)
                if eg >= MIN_GROWTH_PCT_FOR_PEG and pd.notna(pe_for_peg):
                    peg        = float(pe_for_peg) / eg
                    peg_method = "{} EPS growth".format(g_src)
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
            "Rev Q1 (₹Cr)":       (rq1/1e7) if rq1 is not None else None,
            "Rev Q2 (₹Cr)":       (rq2/1e7) if rq2 is not None else None,
            "Rev Q3 (₹Cr)":       (rq3/1e7) if rq3 is not None else None,
            "Rev Q4 (₹Cr)":       (rq4/1e7) if rq4 is not None else None,
            "Rev Growth% (CAGR)": to_num(growth),
        })

    scr = pd.DataFrame(rows)
    if scr.empty:
        return scr

    total_mc = scr["Mkt Cap Raw"].sum()
    scr["MC% of Nifty50"] = (
        scr["Mkt Cap Raw"] / total_mc * 100.0 if total_mc > 0 else None)

    num_cols = ["Price (₹)", "Mkt Cap (₹Cr)", "P/E", "Fwd P/E", "PEG",
                "52W Pos%", "ROIC%", "ROE%", "Int Coverage", "Op Margin%",
                "Debt/Eq", "Quality Score", "Earn Traj", "Momentum Score",
                "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
                "MC% of Nifty50",
                "Rev Q1 (₹Cr)", "Rev Q2 (₹Cr)", "Rev Q3 (₹Cr)", "Rev Q4 (₹Cr)",
                "Rev Growth% (CAGR)"]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA
    scr = compute_conviction_scores(scr)
    return scr


# ── KPI panel ─────────────────────────────────────────────────────────────────
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

    is_all    = (sector_sel == "All Sectors")
    label     = "All Sectors (Nifty 50)" if is_all else sector_sel
    total_mc  = scr["Mkt Cap Raw"].sum()
    sdata     = scr.copy() if is_all else scr[scr["Sector"] == sector_sel]
    sector_mc = sdata["Mkt Cap Raw"].sum()
    pct       = (100.0 if is_all
                 else (sector_mc / total_mc * 100.0 if total_mc > 0 else 0.0))

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
    c1.markdown(_kpi("Sector Mkt Cap",   fmt_mc_inr(sector_mc), "sector total"),         unsafe_allow_html=True)
    c2.markdown(_kpi("Nifty 50 Mkt Cap", fmt_mc_inr(total_mc),  "all 50 stocks"),        unsafe_allow_html=True)
    c3.markdown(_kpi("Sector Share",     "{:.1f}%".format(pct), "{} stocks".format(len(sdata))), unsafe_allow_html=True)
    c4.markdown(_kpi("Median P/E → Fwd",
                     "{:.1f}→{:.1f}".format(med_pe, med_fwd)
                     if pd.notna(med_pe) and pd.notna(med_fwd) else "N/A",
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
            for _, row in top3.iterrows()
        )
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
st.set_page_config(page_title="Nifty 50 Screener v2", layout="wide", page_icon="🇮🇳")
st.markdown(
    "<style>div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}</style>",
    unsafe_allow_html=True)

st.markdown("## 🇮🇳 Nifty 50 Fundamental Screener v2")
st.caption(
    "Wikipedia live universe · Yahoo Finance (.NS) · ROIC from quarterly financials · "
    "Earn Traj from FwdEPS/TrailEPS · MC% of Nifty 50 · 5-factor scoring · INR throughout"
)

page_screener, page_reference = st.tabs(["📊 Screener", "📖 About"])

with page_screener:
    col_r, col_t = st.columns([1, 6])
    with col_r:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_t:
        st.caption("Last loaded: {} · Prices: 1hr · Fundamentals: 24hr".format(
            datetime.now().strftime("%I:%M %p")))

    with st.spinner("Loading Nifty 50 universe from Wikipedia..."):
        universe_df = get_nifty50_universe()
    tickers = tuple(universe_df["Ticker"].tolist())

    with st.spinner("Fetching prices for {} tickers...".format(len(tickers))):
        prices = fetch_prices_batch(tickers)

    with st.spinner("Fetching momentum data..."):
        momentum = fetch_momentum_batch(tickers)

    with st.spinner("Fetching Yahoo fundamentals..."):
        yahoo_fundamentals = fetch_yahoo_fundamentals_all(tickers)

    with st.spinner("Fetching quarterly revenue..."):
        rev_map = fetch_last4_revenue_parallel(tickers)

    total_t  = len(tickers)
    has_pe   = sum(1 for t in tickers if yahoo_fundamentals.get(t, {}).get("pe")        is not None)
    has_fwd  = sum(1 for t in tickers if yahoo_fundamentals.get(t, {}).get("fwd_pe")    is not None)
    has_peg  = sum(1 for t in tickers if yahoo_fundamentals.get(t, {}).get("peg")       is not None)
    has_roic = sum(1 for t in tickers if yahoo_fundamentals.get(t, {}).get("roic")      is not None)
    has_et   = sum(1 for t in tickers if yahoo_fundamentals.get(t, {}).get("earn_traj") is not None)

    st.info(
        "Data coverage — "
        "P/E: {}/{} ({:.0f}%) · Fwd P/E: {}/{} ({:.0f}%) · "
        "PEG: {}/{} ({:.0f}%) · ROIC: {}/{} ({:.0f}%) · "
        "Earn Traj: {}/{} ({:.0f}%) · Source: Yahoo Finance (.NS)".format(
            has_pe,   total_t, has_pe   / total_t * 100,
            has_fwd,  total_t, has_fwd  / total_t * 100,
            has_peg,  total_t, has_peg  / total_t * 100,
            has_roic, total_t, has_roic / total_t * 100,
            has_et,   total_t, has_et   / total_t * 100,
        )
    )

    with st.spinner("Building screener table..."):
        scr = build_screener_table(universe_df, prices, yahoo_fundamentals, rev_map, momentum)

    st.markdown("### Filters")
    with st.expander("Valuation & Size", expanded=True):
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        all_sectors = sorted(scr["Sector"].dropna().unique().tolist())
        sector_sel  = fc1.selectbox("Sector", ["All Sectors"] + all_sectors)
        sort_by     = fc2.selectbox("Sort by", [
            "Sector then Rank", "Score high to low", "Conviction high to low",
            "MC% of Nifty50 high to low",
            "Price low to high", "Price high to low", "Mkt Cap high to low",
            "PE low to high", "Fwd PE low to high", "PEG low to high",
            "Quality Score high", "ROIC high to low", "Earn Traj high to low",
            "Momentum Score high", "52W Pos low to high", "Rev Growth high to low",
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
        "Sector then Rank":          (["Sector", "Rank"],     [True, True]),
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
    sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
    filt   = filt.sort_values(sc, ascending=sa, na_position="last")

    st.caption("Showing {} of {} stocks · Sector: {} · Sort: {}".format(
        len(filt), len(scr), sector_sel, sort_by))

    disp = filt.copy()
    for c in ["P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%",
              "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
              "Quality Score", "Momentum Score", "Ret 1Mo%", "Ret 3Mo%",
              "Ret 6Mo%", "Trailing Vol%", "Score", "Conviction Score",
              "Rev Growth% (CAGR)", "MC% of Nifty50",
              "Price (₹)", "Mkt Cap (₹Cr)",
              "Rev Q1 (₹Cr)", "Rev Q2 (₹Cr)", "Rev Q3 (₹Cr)", "Rev Q4 (₹Cr)"]:
        if c in disp.columns:
            disp[c] = disp[c].round(2)

    disp["Quality Flag"] = disp.apply(
        lambda r: quality_flag(r.get("ROIC%"), r.get("ROE%"),
                               r.get("Int Coverage"),
                               r.get("Op Margin%"), r.get("Debt/Eq")), axis=1)
    disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

    COLS = [
        "Ticker", "Sector",
        "Price (₹)", "Mkt Cap (₹Cr)", "MC% of Nifty50",
        "P/E", "Fwd P/E", "PEG", "PEG Method",
        "Earn Traj",
        "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Quality Flag",
        "Momentum Score", "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
        "52W Pos%", "Score", "Conviction Score", "Rank",
        "Rev Q1 (₹Cr)", "Rev Q2 (₹Cr)", "Rev Q3 (₹Cr)", "Rev Q4 (₹Cr)",
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
**PEG:** < 1.0 = potentially undervalued for its growth rate. Only computed when EPS growth ≥ 5%.

**Earn Traj:** (Forward EPS − Trailing EPS) / |Trailing EPS|. Range −1.0 to +1.0.

**MC% of Nifty50:** This stock's share of total Nifty 50 market cap.
""")

with page_reference:
    st.markdown("## About — Nifty 50 Screener v2")
    st.markdown("""
### Data Source
**Yahoo Finance** via NSE `.NS` tickers · Universe from **Wikipedia (live)**

### India-Specific Notes
| Item | Detail |
|------|--------|
| Default tax rate (ROIC) | 25% India corporate |
| Market cap display | ₹Cr (crores) / ₹L Cr (lakh crore) |
| MC% denominator | Sum of all 50 Nifty constituents |
| Universe refresh | Every 24 hours from Wikipedia |

### Scoring Model
`Valuation 25% + Quality 25% + PEG 20% + Earn Traj 15% + Momentum 15%`
""")
