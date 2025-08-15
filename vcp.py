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
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_KEY"  # Replace with your key
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
        
        # Keep only OHLCV
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"Alpha Vantage fetch failed for {ticker}: {e}")
        return None

def fetch_yahoo(ticker, period="6mo"):
    """Fallback to Yahoo Finance."""
    try:
        df = yf.download(ticker, period=period, interval="1d")
        return df
    except:
        return None

def get_stock_data(ticker, period="6mo"):
    df = fetch_alpha_vantage(ticker)
    if df is None or df.empty:
        df = fetch_yahoo(ticker, period)
    return df

# -------------------
# VCP DETECTION
# -------------------
def detect_vcp(df):
    """Return contraction percentages and peak/trough points."""
    df = df.copy()

    # Ensure required columns exist
    if 'max' not in df.columns:
        df['max'] = np.nan
    if 'min' not in df.columns:
        df['min'] = np.nan

    # Find local peaks and troughs
    peak_idx = argrelextrema(df['High'].values, np.greater, order=5)[0]
    trough_idx = argrelextrema(df['Low'].values, np.less, order=5)[0]

    df.iloc[peak_idx, df.columns.get_loc('max')] = df['High'].iloc[peak_idx]
    df.iloc[trough_idx, df.columns.get_loc('min')] = df['Low'].iloc[trough_idx]

    peaks = df.dropna(subset=['max'])
    troughs = df.dropna(subset=['min'])

    contractions = []
    peak_points, trough_points = [], []

    # Make sure to only iterate while we have both peaks and troughs
    for i in range(min(len(peaks), len(troughs))):
        peak_price = peaks.iloc[i]['max']
        trough_price = troughs.iloc[i]['min']
        contraction_pct = (peak_price - trough_price) / peak_price * 100
        contractions.append(round(contraction_pct, 2))
        peak_points.append((peaks.index[i], peak_price))
        trough_points.append((troughs.index[i], trough_price))

    return contractions, peak_points, trough_points


# -------------------
# PLOTTING
# -------------------
def plot_vcp(df, ticker, peak_points, trough_points):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.set_title(f"{ticker} - Volatility Contraction Pattern")
    ax1.set_ylabel("Price")
    
    for date, price in peak_points:
        ax1.scatter(date, price, color='red', marker='^', s=100)
    for date, price in trough_points:
        ax1.scatter(date, price, color='green', marker='v', s=100)
    
    ax1.legend()
    ax1.grid(True)
    
    ax2.bar(df.index, df['Volume'], color='gray')
    ax2.set_ylabel("Volume")
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

# -------------------
# STREAMLIT UI
# -------------------
st.title("📉 Volatility Contraction Pattern (VCP) Screener")
st.write("Enter comma-separated tickers to scan for VCP patterns.")

tickers_input = st.text_area("Tickers", value="NVDA,AAPL,MSFT,TSLA")
period = st.selectbox("Period", ["3mo","6mo","1y"], index=1)

if st.button("Run Screener"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []

    for ticker in tickers:
        df = get_stock_data(ticker, period)
        if df is None or df.empty:
            continue
        
        contractions, peaks, troughs = detect_vcp(df)
        if len(contractions) >= 3:
            # Check for decreasing contraction pattern
            if contractions[-1] < contractions[-2] < contractions[-3]:
                results.append({
                    "Ticker": ticker,
                    "Last Contraction %": contractions[-1],
                    "Data": df,
                    "Peaks": peaks,
                    "Troughs": troughs
                })

    if results:
        df_results = pd.DataFrame([{"Ticker": r["Ticker"], "Last Contraction %": r["Last Contraction %"]} for r in results])
        df_results.sort_values("Last Contraction %", inplace=True)
        st.dataframe(df_results)

        selected_ticker = st.selectbox("Select ticker to view chart", df_results["Ticker"].tolist())
        for r in results:
            if r["Ticker"] == selected_ticker:
                plot_vcp(r["Data"], r["Ticker"], r["Peaks"], r["Troughs"])
                break
    else:
        st.warning("No VCP candidates found for the given tickers and period.")
