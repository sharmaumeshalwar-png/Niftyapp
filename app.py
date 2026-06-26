import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. PARAMETERS & ROBUST DATA FETCHING
# ==========================================
ticker = "^NSEI"  # Nifty 50 Index
end_date = datetime.date.today()

# SAFETY FIX: Exactly 730 ke bajay 715 days rakhna cloud fetching ko strict rules se bachata hai
start_date = end_date - datetime.timedelta(days=715)

print(f"Fetching 1-Hour Nifty data from {start_date} to {end_date}...")

# CLOUD FIX: Custom headers setup taaki Yahoo Finance blocks ko bypass kiya ja sake
import requests

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
)

# Download with specific session
df = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval="1h",
    auto_adjust=True,
    session=session,  # Adding the session here
)

# Fallback mechanism: Agar data fir bhi empty aaye toh error dene ke bajay retry with close date
if df.empty:
    print("Primary fetch failed, attempting alternative layout...")
    # Thoda data period chota karke re-try karte hain
    start_date_alt = end_date - datetime.timedelta(days=360)
    df = yf.download(
        ticker,
        start=start_date_alt,
        end=end_date,
        interval="1h",
        auto_adjust=True,
        session=session,
    )

if df.empty:
    # Agar ab bhi empty hai toh crash hone ke bajay streamlit window pe warning dikhaye
    import streamlit as st

    st.error(
        "Yahoo Finance API block ho gayi hai Streamlit server par. Kripya app ko refresh karein ya thodi der baad try karein."
    )
    st.stop()

# Multi-index headers ko target karke drop karna
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()

# Baki ka Kalman filter code iske niche as-is chalega...
