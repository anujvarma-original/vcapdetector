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
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_date(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def wiki_to_yahoo_symbol(sym: str) -> str:
    return sym.replace(".", "-")

def last_sma(values: List[float], window: int) -> Optional[float]:
    if values is None or len(values) < window:
        return None
    return float(sum(values[-window:]) / window)

def trim_to_period(data: Dict[str, List], period_days: int) -> Optional[Dict[str, List]]:
    """Trim OHLCV dict to last period_days days."""
    if not data or not data.get("dates"):
        return None
    cutoff = data["dates"][-1].timestamp() - period_days * 86400
    keep_idx = [i for i, d in enumerate(data["dates"]) if d.timestamp() >= cutoff]
    if not keep_idx:
        return None
    for k in data:
        data[k] = [data[k][i] for i in keep_idx]
    return data

# -------------------
# DATA FETCH
# -------------------
def fetch_alpha_vantage(symbol: str, outsize: str = "full") -> Optional[Dict[str, List]]:
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
            except:
                continue
        recs.sort(key=lambda x: x[0])
        if not recs:
            return None
        dates, o, h, l, c, vol = zip(*recs)
        return {"dates": list(dates), "open": list(o), "high": list(h), "low": list(l), "close": list(c), "volume": list(vol)}
    except:
        return None

def fetch_yahoo_json(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[Dict[str, List]]:
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
        dates, o, h, l, c, v = [], [], [], [], [], []
        for i in range(len(ts)):
            vals = (quotes["open"][i], quotes["high"][i], quotes["low"][i], quotes["close"][i], quotes["volume"][i])
            if None in vals:
                continue
            dates.append(to_date(ts[i]))
            o.append(float(vals[0])); h.append(float(vals[1])); l.append(float(vals[2]))
            c.append(float(vals[3])); v.append(float(vals[4]))
        if not dates:
            return None
        return {"dates": dates, "open": o, "high": h, "low": l, "close": c, "volume": v}
    except:
        return None

def fetch_stooq(symbol: str) -> Optional[Dict[str, List]]:
    """Fetch daily OHLCV from Stooq.com (no API key required)."""
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        r = requests.get(url, timeout=10)
        lines = r.text.strip().split("\n")
        if len(lines) <= 1:
            return None
        dates, o, h, l, c, vol = [], [], [], [], [], []
        for row in lines[1:]:
            try:
                d, op, hi, lo, cl, v = row.split(",")
                dates.append(parse_date_ymd(d))
                o.append(float(op)); h.append(float(hi)); l.append(float(lo))
                c.append(float(cl)); vol.append(float(v))
            except:
                continue
        if not dates:
            return None
        return {"dates": dates, "open": o, "high": h, "low": l, "close": c, "volume": vol}
    except:
        return None

@st.cache_data(show_spinner=False)
def get_spy_data(period_days=365, prefer_alpha=True):
    """Get SPY data with triple fallback: Alpha -> Yahoo -> Stooq."""
    spy = None
    if prefer_alpha and ALPHA_VANTAGE_API_KEY:
        spy = fetch_alpha_vantage("SPY", outsize="full")
        if spy:
            return trim_to_period(spy, period_days)
    if spy is None:
        spy = fetch_yahoo_json("SPY", period="1y", interval="1d")
        if spy:
            return trim_to_period(spy, period_days)
    if spy is None:
        spy = fetch_stooq("SPY")
        if spy:
            return trim_to_period(spy, period_days)
    return spy

@st.cache_data(show_spinner=False)
def get_stock_data(symbol: str, period_days: int = 365, prefer_alpha: bool = True) -> Optional[Dict[str, List]]:
    data = None
    if prefer_alpha:
        data = fetch_alpha_vantage(symbol, outsize="full")
        if data is None:
            data = fetch_yahoo_json(wiki_to_yahoo_symbol(symbol), period="1y", interval="1d")
    else:
        data = fetch_yahoo_json(wiki_to_yahoo_symbol(symbol), period="1y", interval="1d")
        if data is None:
            data = fetch_alpha_vantage(symbol, outsize="full")
    return trim_to_period(data, period_days) if data else None

# -------------------
# S&P 500 scrape
# -------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers_and_sectors() -> List[Tuple[str, str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table", {"class": "wikitable"})
    rows = table.find_all("tr")
    headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
    sym_idx, sec_idx = headers.index("Symbol"), headers.index("GICS Sector")
    out = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) <= max(sym_idx, sec_idx):
            continue
        sym = tds[sym_idx].get_text(strip=True).upper()
        sec = tds[sec_idx].get_text(strip=True)
        out.append((sym, sec))
    return out

# -------------------
# RS & filters
# -------------------
def align_and_ratio(stock: Dict[str, List], spy: Dict[str, List]) -> List[float]:
    spy_map = {d.date(): spy["close"][i] for i, d in enumerate(spy["dates"])}
    return [stock["close"][i] / spy_map[d.date()] for i, d in enumerate(stock["dates"]) if d.date() in spy_map and spy_map[d.date()] != 0]

def relative_strength_ok(stock: Dict[str, List], spy: Dict[str, List]) -> bool:
    rs = align_and_ratio(stock, spy)
    if len(rs) < 50:
        return False
    rs_ma50 = last_sma(rs, 50)
    return rs[-1] > rs_ma50 if rs_ma50 else False

def volume_dry_up(stock: Dict[str, List]) -> bool:
    vols = stock["volume"]
    if len(vols) < 50:
        return False
    return (sum(vols[-10:]) / 10.0) < 0.5 * max(vols[-50:])

# -------------------
# VCP detection
# -------------------
def local_peaks(values: List[float], order: int = 5) -> List[int]:
    return [i for i in range(order, len(values) - order)
            if values[i] > max(values[i - order:i] + values[i + 1:i + 1 + order])]

def local_troughs(values: List[float], order: int = 5) -> List[int]:
    return [i for i in range(order, len(values) - order)
            if values[i] < min(values[i - order:i] + values[i + 1:i + 1 + order])]

def compute_contractions(dates: List[datetime], highs: List[float], lows: List[float], order: int = 5):
    p_idx = local_peaks(highs, order)
    t_idx = local_troughs(lows, order)
    contractions, peak_pts, trough_pts = [], [], []
    t_iter = iter(t_idx)
    t_next = next(t_iter, None)
    for pi in p_idx:
        while t_next is not None and t_next < pi:
            t_next = next(t_iter, None)
        if t_next is None:
            break
        pct = (highs[pi] - lows[t_next]) / highs[pi] * 100.0
        contractions.append(round(pct, 2))
        peak_pts.append((dates[pi], highs[pi]))
        trough_pts.append((dates[t_next], lows[t_next]))
    return contractions, peak_pts, trough_pts

def contractions_meet_criteria(contractions: List[float]) -> bool:
    if len(contractions) < 3:
        return False
    last3 = contractions[-3:]
    dec_pairs = sum(1 for a, b in zip(last3, last3[1:]) if b < a)
    return dec_pairs >= 2

# -------------------
# Plotting
# -------------------
def plot_vcp(stock, ticker, peaks, troughs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(stock["dates"], stock["close"], label="Close")
    ax1.set_title(f"{ticker} - VCP Pattern")
    for dt, price in peaks:
        ax1.scatter(dt, price, marker="^", s=80)
    for dt, price in troughs:
        ax1.scatter(dt, price, marker="v", s=80)
    ax1.legend(); ax1.grid(True)
    ax2.bar(stock["dates"], stock["volume"])
    ax2.set_ylabel("Volume"); ax2.grid(True)
    st.pyplot(fig)

# -------------------
# Streamlit UI
# -------------------
st.title("📉 VCP Screener (No pandas) — Triple SPY Fallback")
prefer_alpha = st.toggle("Prefer Alpha Vantage", value=True)
user_tickers_input = st.text_area("Additional Tickers", value="NVDA,AAPL,MSFT,TSLA")
run_btn = st.button("Run Screener")

if run_btn:
    spy = get_spy_data(period_days=365, prefer_alpha=prefer_alpha)
    if spy is None:
        st.error("Could not fetch SPY data from Alpha, Yahoo, or Stooq.")
        st.stop()

    sp_list = get_sp500_tickers_and_sectors()
    if not sp_list:
        st.error("Could not load S&P 500 tickers.")
        st.stop()

    # Strong sector calculation
    sector_strength = {}
    for sector in sorted(set(sec for _, sec in sp_list)):
        tickers_in_sector = [sym for sym, sec in sp_list if sec == sector]
        ratios = []
        for sym in tickers_in_sector:
            sdata = get_stock_data(sym, period_days=365, prefer_alpha=prefer_alpha)
            if not sdata or len(sdata["close"]) < 50:
                continue
            rs_series = align_and_ratio(sdata, spy)
            if len(rs_series) < 50:
                continue
            rs_ma50 = last_sma(rs_series, 50)
            if rs_ma50:
                ratios.append(rs_series[-1] / rs_ma50)
        sector_strength[sector] = float(np.mean(ratios)) if ratios else 0.0

    strong_sectors = [sec for sec, val in sector_strength.items() if val > 1.0]

    rejected, strong_stocks = [], []
    for sym, sec in sp_list:
        if sec not in strong_sectors:
            rejected.append((sym, "WeakSector"))
            continue
        sdata = get_stock_data(sym, period_days=365, prefer_alpha=prefer_alpha)
        if not sdata:
            rejected.append((sym, "NoData"))
            continue
        if not relative_strength_ok(sdata, spy):
            rejected.append((sym, "WeakRS"))
            continue
        strong_stocks.append(sym)

    scan_syms = sorted(set([s.strip().upper() for s in user_tickers_input.split(",") if s.strip()] + strong_stocks))
    results = []
    for sym in scan_syms:
        sdata = get_stock_data(sym, period_days=365, prefer_alpha=prefer_alpha)
        if not sdata:
            rejected.append((sym, "NoData"))
            continue
        cons, peaks, troughs = compute_contractions(sdata["dates"], sdata["high"], sdata["low"], order=5)
        if not contractions_meet_criteria(cons):
            rejected.append((sym, "NoVCP"))
            continue
        if not volume_dry_up(sdata):
            rejected.append((sym, "NoVDU"))
            continue
        results.append({"Ticker": sym, "LastContractionPct": cons[-1], "Peaks": peaks, "Troughs": troughs, "Data": sdata})

    if results:
        st.subheader("Candidates")
        md = "| Ticker | Last Contraction % |\n|---|---|\n"
        for r in sorted(results, key=lambda x: x["LastContractionPct"]):
            md += f"| {r['Ticker']} | {round(r['LastContractionPct'], 2)} |\n"
        st.markdown(md)
        sel = st.selectbox("Select ticker to view chart", [r["Ticker"] for r in results])
        pick = next((r for r in results if r["Ticker"] == sel), None)
        if pick:
            plot_vcp(pick["Data"], pick["Ticker"], pick["Peaks"], pick["Troughs"])
    else:
        st.info("No VCP candidates found.")

    with st.expander("Show rejected tickers with reasons"):
        if rejected:
            st.write("**Reasons:** WeakSector, WeakRS, NoData, NoVCP, NoVDU")
            st.markdown("\n".join(f"- {sym}: {reason}" for sym, reason in rejected))
        else:
            st.write("No rejections logged.")
