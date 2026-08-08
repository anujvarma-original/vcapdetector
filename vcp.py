import math
import streamlit as st
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

# ============================================================
# CONFIG
# ============================================================
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
ALPHA_URL = "https://www.alphavantage.co/query"
YAHOO1_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO2_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"

DEFAULT_PERIOD_DAYS = 365
DEFAULT_VCP_LOOKBACK = 180
USER_AGENT = "Mozilla/5.0 (compatible; VCP-Screener/2.0)"

# ============================================================
# UTILITIES
# ============================================================
def parse_date_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def to_date(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def wiki_to_yahoo_symbol(sym: str) -> str:
    return sym.replace(".", "-")


def safe_mean(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def sma(values: List[float], window: int) -> Optional[float]:
    if values is None or len(values) < window:
        return None
    return safe_mean(values[-window:])


def pct_change(start: float, end: float) -> Optional[float]:
    if start is None or end is None or start == 0:
        return None
    return (end / start - 1.0) * 100.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def trim_to_period(data: Dict[str, List], period_days: int) -> Optional[Dict[str, List]]:
    if not data or not data.get("dates"):
        return None

    cutoff = data["dates"][-1].timestamp() - period_days * 86400
    keep_idx = [i for i, d in enumerate(data["dates"]) if d.timestamp() >= cutoff]
    if not keep_idx:
        return None

    trimmed = {}
    for k, values in data.items():
        trimmed[k] = [values[i] for i in keep_idx]
    return trimmed


def validate_ohlcv(data: Optional[Dict[str, List]], min_bars: int = 60) -> bool:
    if not data:
        return False
    needed = ["dates", "open", "high", "low", "close", "volume"]
    if any(k not in data for k in needed):
        return False
    lengths = [len(data[k]) for k in needed]
    return min(lengths) >= min_bars and len(set(lengths)) == 1


# ============================================================
# DATA FETCHERS
# ============================================================
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
        r.raise_for_status()
        payload = r.json()
        ts = payload.get("Time Series (Daily)")
        if not ts:
            return None

        recs = []
        for d, v in ts.items():
            try:
                recs.append(
                    (
                        parse_date_ymd(d),
                        float(v["1. open"]),
                        float(v["2. high"]),
                        float(v["3. low"]),
                        float(v["4. close"]),
                        float(v["6. volume"]),
                    )
                )
            except Exception:
                continue

        recs.sort(key=lambda x: x[0])
        if not recs:
            return None

        dates, o, h, l, c, vol = zip(*recs)
        return {
            "dates": list(dates),
            "open": list(o),
            "high": list(h),
            "low": list(l),
            "close": list(c),
            "volume": list(vol),
        }
    except Exception:
        return None


def fetch_yahoo_json(
    symbol: str,
    base_url: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[Dict[str, List]]:
    try:
        url = base_url.format(symbol=symbol)
        params = {
            "range": period,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        payload = r.json()
        result = payload.get("chart", {}).get("result")
        if not result:
            return None

        result = result[0]
        ts = result.get("timestamp") or []
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        if not ts or not quotes:
            return None

        dates, o, h, l, c, v = [], [], [], [], [], []
        opens = quotes.get("open", [])
        highs = quotes.get("high", [])
        lows = quotes.get("low", [])
        closes = quotes.get("close", [])
        volumes = quotes.get("volume", [])

        usable = min(len(ts), len(opens), len(highs), len(lows), len(closes), len(volumes))
        for i in range(usable):
            vals = (opens[i], highs[i], lows[i], closes[i], volumes[i])
            if None in vals:
                continue
            try:
                dates.append(to_date(int(ts[i])))
                o.append(float(vals[0]))
                h.append(float(vals[1]))
                l.append(float(vals[2]))
                c.append(float(vals[3]))
                v.append(float(vals[4]))
            except Exception:
                continue

        if not dates:
            return None

        return {
            "dates": dates,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        }
    except Exception:
        return None


def fetch_stooq(symbol: str) -> Optional[Dict[str, List]]:
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if len(lines) <= 1:
            return None

        dates, o, h, l, c, vol = [], [], [], [], [], []
        for row in lines[1:]:
            try:
                parts = row.strip().split(",")
                if len(parts) < 6:
                    continue
                d, op, hi, lo, cl, vv = parts[:6]
                dates.append(parse_date_ymd(d))
                o.append(float(op))
                h.append(float(hi))
                l.append(float(lo))
                c.append(float(cl))
                vol.append(float(vv))
            except Exception:
                continue

        if not dates:
            return None

        return {
            "dates": dates,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol,
        }
    except Exception:
        return None


def fetch_with_fallbacks(symbol: str, period_days: int, prefer_alpha: bool) -> Tuple[Optional[Dict[str, List]], str]:
    """
    Four-level real-market-data fallback chain.

    If Prefer Alpha Vantage is ON:
      1) Alpha Vantage
      2) Yahoo query1
      3) Yahoo query2
      4) Stooq

    If OFF:
      1) Yahoo query1
      2) Yahoo query2
      3) Stooq
      4) Alpha Vantage (if key exists)
    """
    yahoo_symbol = wiki_to_yahoo_symbol(symbol)

    if prefer_alpha:
        attempts = [
            ("Alpha Vantage", lambda: fetch_alpha_vantage(symbol, outsize="full")),
            ("Yahoo-1", lambda: fetch_yahoo_json(yahoo_symbol, YAHOO1_CHART_URL)),
            ("Yahoo-2", lambda: fetch_yahoo_json(yahoo_symbol, YAHOO2_CHART_URL)),
            ("Stooq", lambda: fetch_stooq(symbol)),
        ]
    else:
        attempts = [
            ("Yahoo-1", lambda: fetch_yahoo_json(yahoo_symbol, YAHOO1_CHART_URL)),
            ("Yahoo-2", lambda: fetch_yahoo_json(yahoo_symbol, YAHOO2_CHART_URL)),
            ("Stooq", lambda: fetch_stooq(symbol)),
            ("Alpha Vantage", lambda: fetch_alpha_vantage(symbol, outsize="full")),
        ]

    for source_name, fetcher in attempts:
        data = fetcher()
        if data:
            data = trim_to_period(data, period_days)
            if validate_ohlcv(data, min_bars=50):
                return data, source_name

    return None, "None"


@st.cache_data(show_spinner=False, ttl=1800)
def get_spy_data(period_days: int = DEFAULT_PERIOD_DAYS, prefer_alpha: bool = False):
    return fetch_with_fallbacks("SPY", period_days, prefer_alpha)


@st.cache_data(show_spinner=False, ttl=1800)
def get_stock_data(symbol: str, period_days: int = DEFAULT_PERIOD_DAYS, prefer_alpha: bool = False):
    return fetch_with_fallbacks(symbol, period_days, prefer_alpha)


# ============================================================
# S&P 500 MEMBERSHIP
# ============================================================
def _sp500_static_fallback() -> List[Tuple[str, str]]:
    # Minimal emergency list only; normal operation scrapes the full S&P 500.
    return [
        ("NVDA", "Information Technology"),
        ("AAPL", "Information Technology"),
        ("MSFT", "Information Technology"),
        ("AMZN", "Consumer Discretionary"),
        ("GOOGL", "Communication Services"),
        ("META", "Communication Services"),
        ("TSLA", "Consumer Discretionary"),
        ("JPM", "Financials"),
        ("XOM", "Energy"),
        ("LLY", "Health Care"),
    ]


@st.cache_data(show_spinner=False, ttl=21600)
def get_sp500_tickers_and_sectors() -> List[Tuple[str, str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        table = (
            soup.find("table", {"id": "constituents"})
            or soup.select_one("table.wikitable.sortable")
            or soup.select_one("table.wikitable")
        )
        if not table:
            return _sp500_static_fallback()

        rows = table.find_all("tr")
        if not rows:
            return _sp500_static_fallback()

        headers = [th.get_text(strip=True).lower() for th in rows[0].find_all("th")]

        def find_idx(predicate):
            for i, header in enumerate(headers):
                if predicate(header):
                    return i
            return -1

        sym_idx = find_idx(lambda h: "symbol" in h or "ticker" in h)
        sec_idx = find_idx(lambda h: ("gics" in h and "sector" in h) or h == "sector")

        if sym_idx == -1 or sec_idx == -1:
            return _sp500_static_fallback()

        out: List[Tuple[str, str]] = []
        for tr in rows[1:]:
            tds = tr.find_all("td")
            if len(tds) <= max(sym_idx, sec_idx):
                continue
            sym = tds[sym_idx].get_text(strip=True).upper()
            sec = tds[sec_idx].get_text(strip=True)
            if sym and sec:
                out.append((sym, sec))

        return out if out else _sp500_static_fallback()
    except Exception:
        return _sp500_static_fallback()


# ============================================================
# RELATIVE STRENGTH / TREND
# ============================================================
def align_and_ratio(stock: Dict[str, List], spy: Dict[str, List]) -> List[float]:
    spy_map = {d.date(): spy["close"][i] for i, d in enumerate(spy["dates"])}
    ratios = []
    for i, d in enumerate(stock["dates"]):
        spy_close = spy_map.get(d.date())
        if spy_close and spy_close != 0:
            ratios.append(stock["close"][i] / spy_close)
    return ratios


def relative_strength_metrics(stock: Dict[str, List], spy: Dict[str, List]) -> Dict[str, Any]:
    rs = align_and_ratio(stock, spy)
    if len(rs) < 50:
        return {"ok": False, "ratio": 0.0, "trend_pct": 0.0, "score": 0.0}

    rs50 = safe_mean(rs[-50:])
    rs20 = safe_mean(rs[-20:])
    if not rs50 or not rs20:
        return {"ok": False, "ratio": 0.0, "trend_pct": 0.0, "score": 0.0}

    ratio = rs[-1] / rs50
    trend_pct = pct_change(rs[-20], rs[-1]) or 0.0

    # 0-15 points: above RS50 and rising over the last month.
    score = 0.0
    score += clamp((ratio - 0.98) / 0.07, 0.0, 1.0) * 9.0
    score += clamp((trend_pct + 2.0) / 10.0, 0.0, 1.0) * 6.0

    return {
        "ok": ratio > 1.0,
        "ratio": ratio,
        "trend_pct": trend_pct,
        "score": score,
    }


def trend_template_metrics(stock: Dict[str, List]) -> Dict[str, Any]:
    closes = stock["close"]
    if len(closes) < 200:
        return {"ok": False, "score": 0.0, "sma50": None, "sma150": None, "sma200": None}

    sma50 = sma(closes, 50)
    sma150 = sma(closes, 150)
    sma200 = sma(closes, 200)
    current = closes[-1]

    if not sma50 or not sma150 or not sma200:
        return {"ok": False, "score": 0.0, "sma50": sma50, "sma150": sma150, "sma200": sma200}

    # Minervini-like trend template subset.
    conditions = [
        current > sma50,
        current > sma150,
        current > sma200,
        sma50 > sma150,
        sma150 > sma200,
    ]
    passed = sum(1 for x in conditions if x)
    score = passed / len(conditions) * 15.0

    return {
        "ok": passed >= 4,
        "score": score,
        "sma50": sma50,
        "sma150": sma150,
        "sma200": sma200,
    }


# ============================================================
# VOLUME DRY-UP
# ============================================================
def volume_dry_up_metrics(stock: Dict[str, List]) -> Dict[str, Any]:
    vols = stock["volume"]
    if len(vols) < 50:
        return {
            "ok": False,
            "recent_avg": 0.0,
            "baseline_avg": 0.0,
            "ratio": 1.0,
            "dry_up_pct": 0.0,
            "score": 0.0,
        }

    # Compare recent 10-day average volume to the preceding 40-day average,
    # rather than to one unusually high-volume day.
    recent_avg = safe_mean(vols[-10:]) or 0.0
    baseline_avg = safe_mean(vols[-50:-10]) or 0.0

    if baseline_avg <= 0:
        ratio = 1.0
    else:
        ratio = recent_avg / baseline_avg

    dry_up_pct = max(0.0, (1.0 - ratio) * 100.0)

    # Strong VDU: recent volume <= 65% of prior baseline.
    ok = ratio <= 0.65

    # 0-15 points. Full credit at <= 50% of baseline.
    score = clamp((0.90 - ratio) / 0.40, 0.0, 1.0) * 15.0

    return {
        "ok": ok,
        "recent_avg": recent_avg,
        "baseline_avg": baseline_avg,
        "ratio": ratio,
        "dry_up_pct": dry_up_pct,
        "score": score,
    }


# ============================================================
# VCP DETECTION
# ============================================================
def local_peaks(values: List[float], order: int = 4) -> List[int]:
    out = []
    for i in range(order, len(values) - order):
        left = values[i - order:i]
        right = values[i + 1:i + 1 + order]
        if values[i] >= max(left) and values[i] > max(right):
            out.append(i)
    return out


def local_troughs(values: List[float], order: int = 4) -> List[int]:
    out = []
    for i in range(order, len(values) - order):
        left = values[i - order:i]
        right = values[i + 1:i + 1 + order]
        if values[i] <= min(left) and values[i] < min(right):
            out.append(i)
    return out


def compute_contractions(
    dates: List[datetime],
    highs: List[float],
    lows: List[float],
    order: int = 4,
    lookback_bars: int = 180,
) -> Tuple[List[float], List[Tuple[datetime, float]], List[Tuple[datetime, float]], List[Tuple[int, int]]]:
    if len(highs) < order * 2 + 10:
        return [], [], [], []

    start = max(0, len(highs) - lookback_bars)
    sub_dates = dates[start:]
    sub_highs = highs[start:]
    sub_lows = lows[start:]

    peaks = local_peaks(sub_highs, order)
    troughs = local_troughs(sub_lows, order)

    contractions = []
    peak_pts = []
    trough_pts = []
    pairs = []

    # Pair each peak with the first meaningful trough after it and before the next peak.
    for p_pos, pi in enumerate(peaks):
        next_peak = peaks[p_pos + 1] if p_pos + 1 < len(peaks) else len(sub_highs)
        eligible = [ti for ti in troughs if pi < ti < next_peak]
        if not eligible:
            continue

        ti = min(eligible, key=lambda x: sub_lows[x])
        peak_price = sub_highs[pi]
        trough_price = sub_lows[ti]
        if peak_price <= 0 or trough_price <= 0 or trough_price >= peak_price:
            continue

        pct = (peak_price - trough_price) / peak_price * 100.0

        # Ignore tiny noise and catastrophic drops that are not useful VCP contractions.
        if pct < 2.0 or pct > 50.0:
            continue

        contractions.append(round(pct, 2))
        peak_pts.append((sub_dates[pi], peak_price))
        trough_pts.append((sub_dates[ti], trough_price))
        pairs.append((start + pi, start + ti))

    return contractions, peak_pts, trough_pts, pairs


def contraction_quality(contractions: List[float]) -> Dict[str, Any]:
    if len(contractions) < 2:
        return {
            "ok": False,
            "used": [],
            "tightening_ratio": 0.0,
            "score": 0.0,
        }

    used = contractions[-4:] if len(contractions) >= 4 else contractions[-3:]
    if len(used) < 2:
        return {"ok": False, "used": used, "tightening_ratio": 0.0, "score": 0.0}

    decreasing_pairs = sum(1 for a, b in zip(used, used[1:]) if b < a)
    total_pairs = len(used) - 1
    tightening_ratio = decreasing_pairs / total_pairs if total_pairs else 0.0

    first = used[0]
    last = used[-1]
    shrink = (first - last) / first if first > 0 else 0.0

    # Candidate if most contraction legs tighten and the final leg is not larger than the first.
    ok = len(used) >= 3 and tightening_ratio >= (2.0 / 3.0) and last < first

    # 0-30 points.
    score = tightening_ratio * 18.0
    score += clamp(shrink / 0.60, 0.0, 1.0) * 8.0

    # Reward a tight final contraction. <=8% is excellent, <=12% good.
    if last <= 8.0:
        score += 4.0
    elif last <= 12.0:
        score += 3.0
    elif last <= 15.0:
        score += 1.5

    return {
        "ok": ok,
        "used": used,
        "tightening_ratio": tightening_ratio,
        "score": min(30.0, score),
    }


# ============================================================
# PIVOT / BREAKOUT
# ============================================================
def pivot_metrics(stock: Dict[str, List], pivot_lookback: int = 20) -> Dict[str, Any]:
    highs = stock["high"]
    closes = stock["close"]
    vols = stock["volume"]

    if len(highs) < pivot_lookback + 2:
        return {
            "pivot": None,
            "distance_pct": None,
            "breakout": False,
            "volume_ratio": 0.0,
            "score": 0.0,
            "status": "Developing",
        }

    # Pivot is prior resistance, excluding today's bar.
    prior_highs = highs[-(pivot_lookback + 1):-1]
    pivot = max(prior_highs)
    last_close = closes[-1]
    distance_pct = (last_close / pivot - 1.0) * 100.0 if pivot else None

    avg_vol_20 = safe_mean(vols[-21:-1]) or 0.0
    volume_ratio = (vols[-1] / avg_vol_20) if avg_vol_20 > 0 else 0.0
    breakout = bool(last_close > pivot and volume_ratio >= 1.40)

    # 0-25 points based on proximity plus breakout confirmation.
    score = 0.0
    status = "Developing"

    if distance_pct is not None:
        if breakout:
            score = 25.0
            status = "Breakout"
        elif -1.0 <= distance_pct <= 0.0:
            score = 22.0
            status = "Ready"
        elif -3.0 <= distance_pct < -1.0:
            score = 19.0
            status = "Ready"
        elif -5.0 <= distance_pct < -3.0:
            score = 14.0
            status = "Setup"
        elif -8.0 <= distance_pct < -5.0:
            score = 9.0
            status = "Setup"
        elif 0.0 < distance_pct <= 3.0:
            # Above pivot but not enough breakout volume.
            score = 16.0
            status = "Above Pivot / Low Volume"
        else:
            score = 4.0
            status = "Developing"

    return {
        "pivot": pivot,
        "distance_pct": distance_pct,
        "breakout": breakout,
        "volume_ratio": volume_ratio,
        "score": score,
        "status": status,
    }


# ============================================================
# COMPLETE VCP ANALYSIS
# ============================================================
def analyze_vcp(
    stock: Dict[str, List],
    spy: Dict[str, List],
    order: int,
    lookback_bars: int,
    pivot_lookback: int,
) -> Dict[str, Any]:
    contractions, peaks, troughs, pairs = compute_contractions(
        stock["dates"],
        stock["high"],
        stock["low"],
        order=order,
        lookback_bars=lookback_bars,
    )

    cq = contraction_quality(contractions)
    vdu = volume_dry_up_metrics(stock)
    rs = relative_strength_metrics(stock, spy)
    trend = trend_template_metrics(stock)
    pivot = pivot_metrics(stock, pivot_lookback=pivot_lookback)

    score = cq["score"] + vdu["score"] + rs["score"] + trend["score"] + pivot["score"]
    score = round(clamp(score, 0.0, 100.0), 1)

    # Status is primarily pivot-driven, but only promote mature patterns.
    status = pivot["status"]
    if not cq["ok"]:
        status = "Developing"
    elif cq["ok"] and not vdu["ok"] and status in ("Ready", "Breakout"):
        status = "Setup"

    return {
        "score": score,
        "status": status,
        "contractions": contractions,
        "used_contractions": cq["used"],
        "contraction_ok": cq["ok"],
        "peaks": peaks,
        "troughs": troughs,
        "pairs": pairs,
        "vdu_ok": vdu["ok"],
        "vdu_pct": vdu["dry_up_pct"],
        "vdu_ratio": vdu["ratio"],
        "rs_ok": rs["ok"],
        "rs_ratio": rs["ratio"],
        "rs_trend_pct": rs["trend_pct"],
        "trend_ok": trend["ok"],
        "pivot": pivot["pivot"],
        "distance_pct": pivot["distance_pct"],
        "breakout": pivot["breakout"],
        "breakout_volume_ratio": pivot["volume_ratio"],
    }


# ============================================================
# PLOTTING
# ============================================================
def plot_vcp(stock: Dict[str, List], ticker: str, analysis: Dict[str, Any]):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(stock["dates"], stock["close"], label="Close")
    ax1.set_title(f"{ticker} - VCP Pattern | Score {analysis['score']}")

    for dt, price in analysis["peaks"]:
        ax1.scatter(dt, price, marker="^", s=70)

    for dt, price in analysis["troughs"]:
        ax1.scatter(dt, price, marker="v", s=70)

    if analysis.get("pivot"):
        ax1.axhline(analysis["pivot"], linestyle="--", label=f"Pivot {analysis['pivot']:.2f}")

    ax1.legend()
    ax1.grid(True)

    ax2.bar(stock["dates"], stock["volume"])
    ax2.set_ylabel("Volume")
    ax2.grid(True)

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="VCP Screener", layout="wide")
st.title("📉 VCP Screener (No pandas) — Upgraded 4-Level SPY Fallback")
st.caption(
    "Screens for volatility contraction, volume dry-up, relative strength vs SPY, "
    "trend quality, pivot proximity, and breakout confirmation."
)

with st.sidebar:
    st.header("Settings")
    prefer_alpha = st.toggle(
        "Prefer Alpha Vantage",
        value=False,
        help="For large scans Yahoo-first is usually better. Alpha Vantage free tiers can rate-limit quickly.",
    )
    period_days = st.slider("History days", 250, 730, DEFAULT_PERIOD_DAYS, 30)
    vcp_lookback = st.slider("VCP lookback bars", 90, 250, DEFAULT_VCP_LOOKBACK, 10)
    swing_order = st.slider("Swing sensitivity", 3, 8, 4, 1)
    pivot_lookback = st.slider("Pivot lookback bars", 10, 50, 20, 5)
    minimum_score = st.slider("Minimum VCP score", 0, 100, 60, 5)
    require_vdu = st.checkbox("Require volume dry-up", value=False)
    require_rs = st.checkbox("Require RS above 50-day average", value=False)
    require_trend = st.checkbox("Require trend template", value=False)

user_tickers_input = st.text_area(
    "Additional / priority tickers",
    value="NVDA,AAPL,MSFT,TSLA",
    help="These tickers are always scanned even if they are not in a strong S&P 500 sector.",
)

col1, col2 = st.columns([1, 4])
with col1:
    run_btn = st.button("Run Screener", type="primary")
with col2:
    st.caption("The full S&P 500 scan can make many HTTP requests; results are cached for 30 minutes.")


if run_btn:
    with st.spinner("Fetching SPY benchmark data..."):
        spy, spy_source = get_spy_data(period_days=period_days, prefer_alpha=prefer_alpha)

    if spy is None:
        st.error("Could not fetch SPY data from Alpha Vantage, Yahoo-1, Yahoo-2, or Stooq.")
        st.stop()

    st.success(f"SPY benchmark loaded from: {spy_source}")

    sp_list = get_sp500_tickers_and_sectors()
    if not sp_list:
        st.error("Could not load S&P 500 tickers.")
        st.stop()

    # --------------------------------------------------------
    # Load S&P data once. @st.cache_data prevents duplicate fetches.
    # --------------------------------------------------------
    sector_members: Dict[str, List[str]] = {}
    for sym, sec in sp_list:
        sector_members.setdefault(sec, []).append(sym)

    sector_strength: Dict[str, float] = {}
    stock_cache: Dict[str, Tuple[Optional[Dict[str, List]], str]] = {}

    progress = st.progress(0)
    status_box = st.empty()
    total_sp = len(sp_list)

    # First pass: obtain RS readings used to determine sector strength.
    sector_ratios: Dict[str, List[float]] = {sec: [] for sec in sector_members}

    for idx, (sym, sec) in enumerate(sp_list, start=1):
        status_box.text(f"Loading S&P 500 data: {sym} ({idx}/{total_sp})")
        sdata, source = get_stock_data(sym, period_days=period_days, prefer_alpha=prefer_alpha)
        stock_cache[sym] = (sdata, source)

        if sdata and len(sdata["close"]) >= 50:
            rs_metrics = relative_strength_metrics(sdata, spy)
            if rs_metrics["ratio"] > 0:
                sector_ratios[sec].append(rs_metrics["ratio"])

        progress.progress(idx / max(total_sp, 1))

    for sec, ratios in sector_ratios.items():
        sector_strength[sec] = safe_mean(ratios) or 0.0

    strong_sectors = {sec for sec, value in sector_strength.items() if value > 1.0}

    # --------------------------------------------------------
    # Create scan universe: strong-sector S&P names + user tickers
    # --------------------------------------------------------
    priority_tickers = {
        s.strip().upper()
        for s in user_tickers_input.replace("\n", ",").split(",")
        if s.strip()
    }

    strong_sector_tickers = {sym for sym, sec in sp_list if sec in strong_sectors}
    scan_syms = sorted(priority_tickers | strong_sector_tickers)

    results = []
    rejected = []

    status_box.text(f"Analyzing {len(scan_syms)} VCP candidates...")
    progress.progress(0.0)

    for idx, sym in enumerate(scan_syms, start=1):
        if sym in stock_cache:
            sdata, data_source = stock_cache[sym]
        else:
            sdata, data_source = get_stock_data(sym, period_days=period_days, prefer_alpha=prefer_alpha)
            stock_cache[sym] = (sdata, data_source)

        if not sdata:
            rejected.append((sym, "NoData"))
            progress.progress(idx / max(len(scan_syms), 1))
            continue

        analysis = analyze_vcp(
            sdata,
            spy,
            order=swing_order,
            lookback_bars=vcp_lookback,
            pivot_lookback=pivot_lookback,
        )

        reasons = []
        if not analysis["contraction_ok"]:
            reasons.append("NoVCP")
        if require_vdu and not analysis["vdu_ok"]:
            reasons.append("NoVDU")
        if require_rs and not analysis["rs_ok"]:
            reasons.append("WeakRS")
        if require_trend and not analysis["trend_ok"]:
            reasons.append("WeakTrend")
        if analysis["score"] < minimum_score:
            reasons.append("LowScore")

        if reasons:
            rejected.append((sym, ",".join(reasons)))
            progress.progress(idx / max(len(scan_syms), 1))
            continue

        used_cons = analysis["used_contractions"]
        cons_text = " → ".join(f"{x:.1f}%" for x in used_cons) if used_cons else "—"

        results.append(
            {
                "Ticker": sym,
                "Score": analysis["score"],
                "Status": analysis["status"],
                "Contractions": cons_text,
                "VDU %": round(analysis["vdu_pct"], 1),
                "RS vs 50D": round(analysis["rs_ratio"], 3),
                "Pivot": round(analysis["pivot"], 2) if analysis["pivot"] else None,
                "Distance %": round(analysis["distance_pct"], 2) if analysis["distance_pct"] is not None else None,
                "Today Vol / 20D": round(analysis["breakout_volume_ratio"], 2),
                "Breakout": "YES" if analysis["breakout"] else "",
                "Source": data_source,
                "Analysis": analysis,
                "Data": sdata,
            }
        )
        progress.progress(idx / max(len(scan_syms), 1))

    progress.empty()
    status_box.empty()

    # --------------------------------------------------------
    # Sector strength display
    # --------------------------------------------------------
    with st.expander("Sector relative strength vs SPY"):
        sector_rows = sorted(sector_strength.items(), key=lambda x: x[1], reverse=True)
        md = "| Sector | RS Ratio | Strong? |\n|---|---:|---|\n"
        for sec, val in sector_rows:
            md += f"| {sec} | {val:.3f} | {'YES' if val > 1.0 else ''} |\n"
        st.markdown(md)

    # --------------------------------------------------------
    # Candidate table
    # --------------------------------------------------------
    if results:
        results.sort(key=lambda x: (-x["Score"], abs(x["Distance %"] or 999)))

        st.subheader(f"VCP Candidates ({len(results)})")

        md = (
            "| Ticker | Score | Status | Contractions | VDU % | RS/50D | Pivot | Dist. % | Vol/20D | Breakout |\n"
            "|---|---:|---|---|---:|---:|---:|---:|---:|---|\n"
        )
        for r in results:
            pivot_text = f"${r['Pivot']:.2f}" if r["Pivot"] is not None else "—"
            dist_text = f"{r['Distance %']:.2f}%" if r["Distance %"] is not None else "—"
            md += (
                f"| {r['Ticker']} | {r['Score']:.1f} | {r['Status']} | {r['Contractions']} | "
                f"{r['VDU %']:.1f} | {r['RS vs 50D']:.3f} | {pivot_text} | {dist_text} | "
                f"{r['Today Vol / 20D']:.2f}x | {r['Breakout']} |\n"
            )
        st.markdown(md)

        selectable = [r["Ticker"] for r in results]
        sel = st.selectbox("Select ticker to view chart", selectable)
        pick = next((r for r in results if r["Ticker"] == sel), None)

        if pick:
            a = pick["Analysis"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("VCP Score", f"{a['score']:.1f}/100")
            m2.metric("Status", a["status"])
            m3.metric("Volume Dry-Up", f"{a['vdu_pct']:.1f}%")
            m4.metric("Distance to Pivot", f"{a['distance_pct']:.2f}%" if a["distance_pct"] is not None else "—")
            m5.metric("Breakout Volume", f"{a['breakout_volume_ratio']:.2f}x")

            st.write(
                "**Detected contractions:** "
                + (" → ".join(f"{x:.1f}%" for x in a["used_contractions"]) if a["used_contractions"] else "None")
            )
            plot_vcp(pick["Data"], pick["Ticker"], a)
    else:
        st.info("No candidates met the current score/filter settings.")

    # --------------------------------------------------------
    # Rejection diagnostics
    # --------------------------------------------------------
    with st.expander(f"Rejected tickers ({len(rejected)})"):
        st.write(
            "**Reasons:** NoData, NoVCP, NoVDU, WeakRS, WeakTrend, LowScore. "
            "Multiple reasons may appear for one ticker."
        )
        if rejected:
            st.markdown("\n".join(f"- **{sym}**: {reason}" for sym, reason in rejected))
        else:
            st.write("No rejections logged.")

    st.caption(
        "VCP Score weighting: contractions 30, volume dry-up 15, relative strength 15, "
        "trend template 15, pivot/breakout position 25. Treat the score as a ranking aid, not a trading signal."
    )
