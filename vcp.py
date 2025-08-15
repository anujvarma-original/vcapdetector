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
    df.loc[df.index[trough_idx], "min"] = df["Low"].iloc[t]()_]()]()
