# vcp.py  (no pandas version)

import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict

# -------------------
# CONFIG
# -------------------
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
ALPHA_URL = "https://www.alphavantage.co/query"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# -------------------
# UTILITIES
# -------------------
def parse_date_ymd(s: str) -> datetime:
    # Alpha Vantage returns YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_date(ts: int) -> datetime:
    # Yahoo returns UNIX seconds
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def wiki_to_yahoo_symbol(sym: str) -> str:
    # BRK.B -> BRK-B; BF.B -> BF-B, etc.
    return sym.replace(".", "-")

def last_sma(values: List[float], window: int) -> Optional[float]:
    if values is None or len(values) < window:
        return None
    return float(sum(values[-window:]) / window)

# -------------------
# DATA FETCH (NO PANDAS)
# -------------------
def fetch_alpha_vantage(symbol: str, outsize: str = "full") -> Optional[Dict[str, List]]:
    """
    Returns dict with keys: dates, open, high, low, close, volume (all lists, ascending by date).
    """
    if not ALPHA_VANTAGE_API_KEY:
        return None
    try:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outsize,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        r = requests.get(ALPHA_URL, params=params, timeout=20)
        data = r.json()
        ts = data.get("Time Series (Daily)")
        if not ts:
            return None

        # Build and sort by date ascending
        recs = []
        for d, v in ts.items():
            try:
                recs.append((
                    parse_date_ymd(d),
                    float(v["1. open"]),
                    float(v["2. high"]),
                    float(v["3. low"]),
                    float(v["4. close"]),
                    float(v["6. volume"])
                ))
            except Exception:
                continue
        recs.sort(key=lambda x: x[0])
        if not recs:
            return None

        dates, o, h, l, c, v = zip(*recs)
        return {"dates": list(dates), "open": list(o), "high": list(h), "low": list(l), "close": list(c), "volume": list(v)}
    except Exception:
        return None

def fetch_yahoo_json(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[Dict[str, List]]:
    """
    Yahoo Chart API (no key). Returns same dict keys as Alpha function.
    """
    try:
        url = YAHOO_CHART_URL.format(symbol=symbol)
        params = {"range": period, "interval": interval, "includePrePost": "false"}
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return None
        result = result[0]
        ts = result.get("timestamp")
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        if not ts or not quotes:
            return None

        opens = quotes.get("open", [])
        highs = quotes.get("high", [])
        lows = quotes.get("low", [])
        closes = quotes.get("close", [])
        vols = quotes.get("volume", [])

        # Filter out None points
        dates, o, h, l, c, v = [], [], [], [], [], []
        for i in range(len(ts)):
            o_i = opens[i] if i < len(opens) else None
            h_i = highs[i] if i < len(highs) else None
            l_i = lows[i] if i < len(lows) else None
            c_i = closes[i] if i < len(closes) else None
            v_i = vols[i] if i < len(vols) else None
            if None in (o_i, h_i, l_i, c_i, v_i):
                continue
            dates.append(to_date(ts[i]))
            o.append(float(o_i)); h.append(float(h_i)); l.append(float(l_i)); c.append(float(c_i)); v.append(float(v_i))

        if not dates:
            return None

        return {"dates": dates, "open": o, "high": h, "low": l, "close": c, "volume": v}
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_stock_data(symbol: str, period_days: int = 365, prefer_alpha: bool = True) -> Optional[Dict[str, List]]:
    """
    Try Alpha Vantage (primary) then Yahoo (fallback), return last `period_days`.
    """
    data = None
    if prefer_alpha:
        data = fetch_alpha_vantage(symbol, outsize="full")
        if data is None:
            data = fetch_yahoo_json(wiki_to_yahoo_symbol(symbol), period="1y", interval="1d")
    else:
        data = fetch_yahoo_json(wiki_to_yahoo_symbol(symbol), period="1y", interval="1d")
        if data is None:
            data = fetch_alpha_vantage(symbol, outsize="full")

    if data is None:
        return None

    # Keep only last `period_days` worth of data
    if data["dates"]:
        cutoff = data["dates"][-1].timestamp() - period_days * 86400
        keep_idx = [i for i, d in enumerate(data["dates"]) if d.timestamp() >= cutoff]
        if len(keep_idx) < 1:
            return None
        # slice
        for k in ["dates", "open", "high", "low", "close", "volume"]:
            data[k] = [data[k][i] for i in keep_idx]
    return data

# -------------------
# S&P 500 SCRAPE (NO PANDAS)
# -------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers_and_sectors() -> List[Tuple[str, str]]:
    """
    Returns list of (symbol, sector) from Wikipedia using BeautifulSoup.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table", {"class": "wikitable"})

    rows = table.find_all("tr")
    out = []
    # Identify column indices by header text
    headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
    try:
        sym_idx = headers.index("Symbol")
        sec_idx = headers.index("GICS Sector")
    except ValueError:
        # Fallback: assume first and fourth columns
        sym_idx, sec_idx = 0, 3

    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) <= max(sym_idx, sec_idx):
            continue
        sym = tds[sym_idx].get_text(strip=True)
        sec = tds[sec_idx].get_text(strip=True)
        if sym:
            out.append((sym.upper(), sec))
    return out

# -------------------
# RELATIVE STRENGTH & FILTERS (NO PANDAS)
# -------------------
def align_and_ratio(stock: Dict[str, List], spy: Dict[str, List]) -> List[float]:
    """
    Align by date and return RS ratio series: stock_close / spy_close
    """
    spy_map = {d.date(): spy["close"][i] for i, d in enumerate(spy["dates"])}
    rs = []
    for i, d in enumerate(stock["dates"]):
        c = stock["close"][i]
        spyc = spy_map.get(d.date())
        if spyc is None or spyc == 0:
            continue
        rs.append(float(c) / float(spyc))
    return rs

def relative_strength_ok(stock: Dict[str, List], spy: Dict[str, List]) -> bool:
    rs = align_and_ratio(stock, spy)
    if len(rs) < 50:
        return False
    rs_last = rs[-1]
    rs_ma50 = last_sma(rs, 50)
    if rs_ma50 is None:
        return False
    return rs_last > rs_ma50

def volume_dry_up(stock: Dict[str, List]) -> bool:
    vols = stock["volume"]
    if len(vols) < 50:
        return False
    recent10 = float(sum(vols[-10:])) / 10.0
    peak50 = float(max(vols[-50:]))
    return recent10 < 0.5 * peak50

# -------------------
# VCP DETECTION (NO PANDAS / NO SCIPY)
# -------------------
def local_peaks(values: List[float], order: int = 5) -> List[int]:
    idxs = []
    n = len(values)
    for i in range(order, n - order):
        left = values[i - order:i]
        right = values[i + 1:i + 1 + order]
        if len(left) == order and len(right) == order:
            if values[i] > max(left + right):
                idxs.append(i)
    return idxs

def local_troughs(values: List[float], order: int = 5) -> List[int]:
    idxs = []
    n = len(values)
    for i in range(order, n - order):
        left = values[i - order:i]
        right = values[i + 1:i + 1 + order]
        if len(left) == order and len(right) == order:
            if values[i] < min(left + right):
                idxs.append(i)
    return idxs

def compute_contractions(dates: List[datetime], highs: List[float], lows: List[float], order: int = 5) -> Tuple[List[float], List[Tuple[datetime, float]], List[Tuple[datetime, float]]]:
    p_idx = local_peaks(highs, order=order)
    t_idx = local_troughs(lows, order=order)
    # Pair peak -> following trough
    contractions = []
    peak_points = []
    trough_points = []
    t_iter = iter(t_idx)
    t_next = next(t_iter, None)
    for pi in p_idx:
        # advance to the first trough after this peak
        while t_next is not None and t_next < pi:
            t_next = next(t_iter, None)
        if t_next is None:
            break
        peak = highs[pi]
        trough = lows[t_next]
        if peak > 0 and trough < peak:
            pct = (peak - trough) / peak * 100.0
            contractions.append(round(pct, 2))
            peak_points.append((dates[pi], peak))
            trough_points.append((dates[t_next], trough))
        # continue to next peak (but do not advance trough yet; next peak will get same or later trough)
    return contractions, peak_points, trough_points

def contractions_meet_criteria(contractions: List[float]) -> bool:
    if len(contractions) < 3:
        return False
    last3 = contractions[-3:]
    dec_pairs = 0
    if last3[1] < last3[0]:
        dec_pairs += 1
    if last3[2] < last3[1]:
        dec_pairs += 1
    return dec_pairs >= 2  # at least 2 of last 3 decreasing

# -------------------
# PLOTTING (NO PANDAS)
# -------------------
def plot_vcp(stock: Dict[str, List], ticker: str, peak_points: List[Tuple[datetime, float]], trough_points: List[Tuple[datetime, float]]):
    dates = stock["dates"]
    closes = stock["close"]
    vols = stock["volume"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(dates, closes, label="Close")
    ax1.set_title(f"{ticker} - Volatility Contraction Pattern")
    ax1.set_ylabel("Price")

    for dt, price in peak_points:
        ax1.scatter(dt, price, marker="^", s=80)
    for dt, price in trough_points:
        ax1.scatter(dt, price, marker="v", s=80)

    ax1.legend(); ax1.grid(True)
    ax2.bar(dates, vols)
    ax2.set_ylabel("Volume"); ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

# -------------------
# STREAMLIT UI
# -------------------
st.title("📉 VCP Screener (No pandas) — Minervini-style")

prefer_alpha = st.toggle("Prefer Alpha Vantage (may be rate-limited)", value=True)
user_tickers_input = st.text_area("Additional Tickers (comma-separated)", value="NVDA,AAPL,MSFT,TSLA")
run_btn = st.button("Run Screener")

if run_btn:
    # Get SPY (RS baseline)
    spy = get_stock_data("SPY", period_days=365, prefer_alpha=prefer_alpha)
    if spy is None:
        st.error("Could not fetch SPY data. Please try again or switch data source preference.")
        st.stop()

    # Get S&P 500 universe
    sp_list = get_sp500_tickers_and_sectors()
    if not sp_list:
        st.error("Could not load S&P 500 tickers.")
        st.stop()

    # Determine strong sectors
    sector_strength: Dict[str, float] = {}
    for sector in sorted(set(sec for _, sec in sp_list)):
        tickers_in_sector = [sym for sym, sec in sp_list if sec == sector]
        ratios = []
        for sym in tickers_in_sector:
            sdata = get_stock_data(sym, period_days=365, prefer_alpha=prefer_alpha)
            if not sdata or len(sdata["close"]) < 50:
                continue
            # RS / RS_MA50
            rs_series = align_and_ratio(sdata, spy)
            if len(rs_series) < 50:
                continue
            rs_last = rs_series[-1]
            rs_ma50 = last_sma(rs_series, 50)
            if rs_ma50 and rs_ma50 != 0:
                ratios.append(rs_last / rs_ma50)
        sector_strength[sector] = float(np.mean(ratios)) if ratios else 0.0

    strong_sectors = [sec for sec, val in sector_strength.items() if val > 1.0]

    # Strong stocks in strong sectors
    strong_stocks: List[str] = []
    for sym, sec in sp_list:
        if sec not in strong_sectors:
            continue
        sdata = get_stock_data(sym, period_days=365, prefer_alpha=prefer_alpha)
        if not sdata:
            continue
        if not relative_strength_ok(sdata, spy):
            continue
        strong_stocks.append(sym)

    # Final scan list = user input + strong stocks
    input_syms = [s.strip().upper() for s in user_tickers_input.split(",") if s.strip()]
    scan_syms = sorted(set(input_syms + strong_stocks))

    # Scan for VCP with VDU
    results = []
    for sym in scan_syms:
        sdata = get_stock_data(sym, period_days=365, prefer_alpha=prefer_alpha)
        if not sdata:
            continue
        cons, peaks, troughs = compute_contractions(sdata["dates"], sdata["high"], sdata["low"], order=5)
        if contractions_meet_criteria(cons) and volume_dry_up(sdata):
            results.append({
                "Ticker": sym,
                "LastContractionPct": cons[-1] if cons else None,
                "Peaks": peaks,
                "Troughs": troughs,
                "Data": sdata
            })

    # Display results WITHOUT pandas
    if results:
        results = [r for r in results if r["LastContractionPct"] is not None]
        results.sort(key=lambda x: x["LastContractionPct"])
        st.subheader("Candidates")
        # Render a simple markdown table
        md = "| Ticker | Last Contraction % |\n|---|---|\n"
        for r in results:
            md += f"| {r['Ticker']} | {round(r['LastContractionPct'], 2)} |\n"
        st.markdown(md)

        # Chart selection
        tickers_for_chart = [r["Ticker"] for r in results]
        sel = st.selectbox("Select ticker to view chart", tickers_for_chart)
        pick = next((r for r in results if r["Ticker"] == sel), None)
        if pick:
            plot_vcp(pick["Data"], pick["Ticker"], pick["Peaks"], pick["Troughs"])
    else:
        st.info("No VCP candidates found under current filters.")

    # Optional: show sector strength
    with st.expander("Show sector strength (RS / RS_MA50 averages)"):
        # Simple markdown list without pandas
        lines = [f"- **{sec}**: {round(val, 3)}" for sec, val in sorted(sector_strength.items(), key=lambda x: x[0])]
        st.markdown("\n".join(lines))
