import datetime
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================================
# 1. PARAMETERS & BYPASS DATA FETCHING
# ==========================================
ticker = "^NSEI"
end_date = int(datetime.datetime.now().timestamp())
start_date = int(
    (
        datetime.datetime.now() - datetime.timedelta(days=715)
    ).timestamp()
)  # 2 Years Timestamp

st.write("🔄 Nifty 1-Hour Data Load Ho Raha Hai (Bypass Mode)...")


# Direct Yahoo Query API Fetch Method (Yfinance library ke bina)
def fetch_data_direct(symbol, start, end):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start}&period2={end}&interval=1h"

    # Browser ki tarah act karne ke liye strong headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()

        # JSON parsing into DataFrame
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        indicators = result["indicators"]["quote"][0]

        ohlc_df = pd.DataFrame(
            {
                "Open": indicators["open"],
                "High": indicators["high"],
                "Low": indicators["low"],
                "Close": indicators["close"],
                "Volume": indicators["volume"],
            },
            index=pd.to_datetime(timestamps, unit="s"),
        )

        # Indian Timezone (IST) me convert karna
        ohlc_df.index = ohlc_df.index.tz_localize("UTC").tz_convert(
            "Asia/Kolkata"
        )
        return ohlc_df.dropna()
    except Exception as e:
        return pd.DataFrame()  # Khali DF return karega agar fail hua toh


# Data Fetch Execute karna
df = fetch_data_direct(ticker, start_date, end_date)

# Fallback: Agar Nifty Index block ho toh temporary testing ke liye Nifty ETF (INDY) fetch karega
if df.empty:
    st.warning(
        "⚠️ Direct Index block hai, Backup Ticker (Nifty ETF) se data try kar rahe hain..."
    )
    df = fetch_data_direct("INDY", start_date, end_date)

if df.empty:
    st.error(
        "❌ Streamlit Servers ko Yahoo ne poori tarah block kiya hai. Kripya app ko 'Reboot' karein Streamlit settings se."
    )
    st.stop()
else:
    st.success(f"✅ Data Successfully Fetched! Total Rows: {len(df)}")

# Ab iske neeche aapka Kalman Filter ka function aur Strategy logic as-is chalega.
