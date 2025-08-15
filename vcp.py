import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import yfinance as yf
from scipy.signal import argrelextrema

# Streamlit config
st.set_page_config(page_title="VCP Pattern Detector", layout="wide")

# Alpha Vantage API key from Streamlit secrets
ALPHA_VANTAGE_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]

# Function to get data from Alpha Vantage
def get_data_alphavantage(ticker, outputsize="compact"):
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={ticker}"
            f"&outputsize={outputsize}"
            f"&apikey={ALPHA_VANTAGE_KEY}"
        )
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()

        if "Time Series (Daily)" not in data:
            raise ValueError("Invalid Alpha Vantage response.")

        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividend", "Split"]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        st.warning(f"[Alpha Vantage] Error fetching {ticker}: {e}")
        return None

# Function to get data from Yahoo Finance
def get_data_yahoo(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d")
        return df
    except Exception as e:
        st.warning(f"[Yahoo Finance] Error fetching {ticker}: {e}")
        return None

# Combined function to get VCP data
def get_vcp_data(ticker, period="6mo"):
    df = get_data_alphavantage(ticker, outputsize="full")
    if df is None or df.empty:
        st.info(f"Falling back to Yahoo Finance for {ticker}...")
        df = get_data_yahoo(ticker, period=period)

    if df is None or df.empty:
        st.error(f"No data available for {ticker}.")
        return None, None, None, None

    df['max'] = df['High'][argrelextrema(df['High'].values, np.greater, order=5)[0]]
    df['min'] = df['Low'][argrelextrema(df['Low'].values, np.less, order=5)[0]]

    peaks = df.dropna(subset=['max'])
    troughs = df.dropna(subset=['min'])

    contractions = []
    peak_points, trough_points = [], []
    for i in range(min(len(peaks), len(troughs))):
        if i < len(troughs):
            peak_price = peaks.iloc[i]['max']
            trough_price = troughs.iloc[i]['min']
            contraction_pct = (peak_price - trough_price) / peak_price * 100
            contractions.append(round(contraction_pct, 2))
            peak_points.append((peaks.index[i], peak_price))
            trough_points.append((troughs.index[i], trough_price))

    return df, contractions, peak_points, trough_points

# Plotting function
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
    return fig

# UI
st.title("📉 Volatility Contraction Pattern (VCP) Detector")

ticker = st.text_input("Enter stock ticker:", "NVDA").upper()

if st.button("Run VCP Analysis"):
    df, contractions, peak_points, trough_points = get_vcp_data(ticker)

    if df is not None:
        st.subheader(f"{ticker} Contractions")
        st.write(pd.DataFrame({"Contraction %": contractions}))

        fig = plot_vcp(df, ticker, peak_points, trough_points)
        st.pyplot(fig)
