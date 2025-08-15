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
    """Fetch daily OHLCV from Alpha Vantage."""
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

def fetch_yahoo(ticker, period="6mo"):
    """Fallback to Yahoo Finance."""
    try:
        df = yf.download(ticker, period=period, interval="1d")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df
    except:
        return None

def get_stock_data(ticker, period="6mo"):
    """Try Alpha Vantage first, then Yahoo Finance."""
    df = fetch_alpha_vantage(ticker)
    if df is None or df.empty:
        df = fetch_yahoo(ticker, period)
    return df

# -------------------
# VCP DETECTION
# -------------------
def detect_vcp(df):
    """Return contraction percentages and peak/trough points without KeyError."""
    if not isinstance(df, pd.DataFrame):
        return [], [], []

    df = df.copy()

    if "High" not in df.columns or "Low" not in df.columns:
        return [], [], []

    # Create empty columns for peaks/troughs
    df["max"] = np.nan
    df["min"] = np.nan

    # Find local peaks/troughs
    peak_idx = argrelextrema(df["High"].values, np.greater, order=5)[0]
    trough_idx = argrelextrema(df["Low"].values, np.less, order=5)[0]

    df.loc[df.index[peak_idx], "max"] = df["High"].iloc[peak_idx]
    df.loc[df.index[trough_idx], "min"] = df["Low"].iloc[trough_idx]

    # Filter without dropna() to avoid KeyError
    peaks = df[df["max"].notna()]
    troughs = df[df["min"].notna()]

    contractions = []
    peak_points, trough_points = [], []

    for i in range(min(len(peaks), len(troughs))):
        peak_price = peaks.iloc[i]["max"]
        trough_price = troughs.iloc[i]["min"]
        contraction_pct = (peak_price - trough_price) / peak_price * 100
        contractions.append(round(contraction_pct, 2))
        peak_points.append((peaks.index[i], peak_price))
        trough_points.append((troughs.index[i], trough_price))

    return contractions, peak_points, trough_points

# -------------------
# PLOTTING
# -------------------
def plot_vcp(df, ticker, peak_points, trough_points):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[3, 1]})

    ax1.plot(df.index, df["Close"], label="Close Price", color="blue")
    ax1.set_title(f"{ticker} - Volatility Contraction Pattern")
    ax1.set_ylabel("Price")

    for date, price in peak_points:
        ax1.scatter(date, price, color="red", marker="^", s=100)
    for date, price in trough_points:
        ax1.scatter(date, price, color="green", marker="v", s=100)

    ax1.legend()
    ax1.grid(True)

    ax2.bar(df.index, df["Volume"], color="gray")
    ax2.set_ylabel("Volume")
    ax2.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# -------------------
# S&P 500 TICKERS
# -------------------
@st.cache_data
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table["Symbol"].tolist()

# -------------------
# STREAMLIT UI
# -------------------
st.title("📉 Volatility Contraction Pattern (VCP) Screener")
st.write("Scans your input tickers + all S&P 500 tickers for VCP patterns.")

tickers_input = st.text_area("Additional Tickers (comma-separated)", value="NVDA,AAPL,MSFT,TSLA")
period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

if st.button("Run Screener"):
    input_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    sp500_tickers = get_sp500_tickers()
    tickers_to_scan = sorted(set(input_tickers + sp500_tickers))

    results = []
    for ticker in tickers_to_scan:
        df = get_stock_data(ticker, period)
        if df is None or df.empty:
            continue

        contractions, peaks, troughs = detect_vcp(df)
        if len(contractions) >= 3:
            if contractions[-1] < contractions[-2] < contractions[-3]:
                results.append({
                    "Ticker": ticker,
                    "Last Contraction %": contractions[-1],
                    "Data": df,
                    "Peaks": peaks,
                    "Troughs": troughs
                })

    if results:
        df_results = pd.DataFrame(
            [{"Ticker": r["Ticker"], "Last Contraction %": r["Last Contraction %"]} for r in results]
        ).sort_values("Last Contraction %")
        st.dataframe(df_results)

        selected_ticker = st.selectbox("Select ticker to view chart", df_results["Ticker"].tolist())
        for r in results:
            if r["Ticker"] == selected_ticker:
                plot_vcp(r["Data"], r["Ticker"], r["Peaks"], r["Troughs"])
                break
    else:
        st.warning("No VCP candidates found.")
