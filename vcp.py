import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import yfinance as yf
from scipy.signal import argrelextrema

# -------------------
# CONFIG
# -------------------
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
BASE_URL = "https://www.alphavantage.co/query"

# -------------------
# DATA FETCH
# -------------------
def fetch_alpha_vantage(ticker):
    try:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        r = requests.get(BASE_URL, params=params)
        data = r.json()
        if "Time Series (Daily)" not in data:
            return None
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "6. volume": "Volume"
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except:
        return None

def fetch_yahoo(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df
    except:
        return None

def get_stock_data(ticker, period="1y"):
    df = fetch_alpha_vantage(ticker)
    if df is None or df.empty:
        df = fetch_yahoo(ticker, period)
    return df

# -------------------
# VCP DETECTION
# -------------------
def detect_vcp(df):
    if not isinstance(df, pd.DataFrame):
        return [], [], []
    if "High" not in df.columns or "Low" not in df.columns:
        return [], [], []

    df = df.copy()
    df["max"] = np.nan
    df["min"] = np.nan

    peak_idx = argrelextrema(df["High"].values, np.greater, order=5)[0]
    trough_idx = argrelextrema(df["Low"].values, np.less, order=5)[0]

    df.loc[df.index[peak_idx], "max"] = df["High"].iloc[peak_idx]
    df.loc[df.index[trough_idx], "min"] = df["Low"].iloc[trough_idx]

    peaks = df[df["max"].notna()]
    troughs = df[df["min"].notna()]

    contractions, peak_points, trough_points = [], [], []
    for i in range(min(len(peaks), len(troughs))):
        peak_price = peaks.iloc[i]["max"]
        trough_price = troughs.iloc[i]["min"]
        contraction_pct = (peak_price - trough_price) / peak_price * 100
        contractions.append(round(contraction_pct, 2))
        peak_points.append((peaks.index[i], peak_price))
        trough_points.append((troughs.index[i], trough_price))
    return contractions, peak_points, trough_points

# -------------------
# FILTERS
# -------------------
def contractions_meet_criteria(contractions):
    if len(contractions) < 3:
        return False
    last3 = contractions[-3:]
    return sum(1 for i in range(1, len(last3)) if last3[i] < last3[i-1]) >= 2

def volume_dry_up(df):
    if len(df) < 50:
        return False
    recent_vol = df["Volume"][-10:].mean()
    peak_vol = df["Volume"][-50:].max()
    return recent_vol < 0.5 * peak_vol

@st.cache_data
def get_sp500_tickers_and_sectors():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table[["Symbol", "GICS Sector"]]

def relative_strength(df, spy_df):
    rs_series = df["Close"] / spy_df["Close"]
    rs_ma50_series = rs_series.rolling(50).mean()

    # Convert to floats to avoid ambiguous truth value
    rs_val = float(rs_series.iloc[-1]) if not pd.isna(rs_series.iloc[-1]) else np.nan
    rs_ma50_val = float(rs_ma50_series.iloc[-1]) if not pd.isna(rs_ma50_series.iloc[-1]) else np.nan

    return rs_val, rs_ma50_val

# -------------------
# PLOTTING
# -------------------
def plot_vcp(df, ticker, peak_points, trough_points):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(df.index, df["Close"], label="Close Price", color="blue")
    ax1.set_title(f"{ticker} - Volatility Contraction Pattern")
    ax1.set_ylabel("Price")
    for date, price in peak_points:
        ax1.scatter(date, price, color="red", marker="^", s=100)
    for date, price in trough_points:
        ax1.scatter(date, price, color="green", marker="v", s=100)
    ax1.legend(); ax1.grid(True)
    ax2.bar(df.index, df["Volume"], color="gray")
    ax2.set_ylabel("Volume"); ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

# -------------------
# STREAMLIT UI
# -------------------
st.title("📉 Volatility Contraction Pattern (VCP) Screener - Minervini Style")
tickers_input = st.text_area("Additional Tickers (comma-separated)", value="NVDA,AAPL,MSFT,TSLA")

if st.button("Run Screener"):
    spy_df = get_stock_data("SPY", "1y")
    sp500_df = get_sp500_tickers_and_sectors()

    # Determine strong sectors
    sector_strength = {}
    for sector in sp500_df["GICS Sector"].unique():
        tickers = sp500_df[sp500_df["GICS Sector"] == sector]["Symbol"].tolist()
        rs_values = []
        for t in tickers:
            df = get_stock_data(t, "1y")
            if df is None or df.empty:
                continue
            rs, rs_ma50 = relative_strength(df, spy_df)
            if not pd.isna(rs) and not pd.isna(rs_ma50):
                rs_values.append(rs / rs_ma50)
        sector_strength[sector] = np.mean(rs_values) if rs_values else 0
    strong_sectors = [s for s, v in sector_strength.items() if v > 1]

    # Strong stocks in strong sectors
    input_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    strong_stocks = []
    for _, row in sp500_df.iterrows():
        if row["GICS Sector"] in strong_sectors:
            df = get_stock_data(row["Symbol"], "1y")
            if df is None or df.empty:
                continue
            rs, rs_ma50 = relative_strength(df, spy_df)
            if not pd.isna(rs) and not pd.isna(rs_ma50) and rs > rs_ma50:
                strong_stocks.append(row["Symbol"])

    tickers_to_scan = sorted(set(input_tickers + strong_stocks))

    # Scan for VCP
    results = []
    for ticker in tickers_to_scan:
        df = get_stock_data(ticker, "1y")
        if df is None or df.empty:
            continue
        contractions, peaks, troughs = detect_vcp(df)
        if contractions_meet_criteria(contractions) and volume_dry_up(df):
            results.append({
                "Ticker": ticker,
                "Last Contraction %": contractions[-1],
                "Data": df,
                "Peaks": peaks,
                "Troughs": troughs
            })

    # SAFE RESULTS DISPLAY
    if results:
        df_results = pd.DataFrame(
            [{"Ticker": r["Ticker"], "Last Contraction %": r["Last Contraction %"]} for r in results]
        ).sort_values("Last Contraction %")

        if not df_results.empty:
            st.dataframe(df_results)
            ticker_list = df_results["Ticker"].tolist()
            if ticker_list:
                selected_ticker = st.selectbox("Select ticker to view chart", ticker_list)
                for r in results:
                    if r["Ticker"] == selected_ticker:
                        plot_vcp(r["Data"], r["Ticker"], r["Peaks"], r["Troughs"])
                        break
            else:
                st.info("No tickers available for chart display.")
        else:
            st.info("No valid results to display.")
    else:
        st.warning("No VCP candidates found under current filters.")
