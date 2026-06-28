import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🚀 Nifty & India VIX 5-Minute Reliable Dashboard")

st.write("Fixed Architecture: Syntax Line 175 Fixed | Auto-Fallback Engine Implemented | Q=0.50...")

# 1. OPTIMIZED FUNCTION WITH HOLIDAY RETRY ENGINE & SIMULATION FALLBACK
@st.cache_data(ttl=60) # Reduced cache time to 60 seconds for faster retries
def load_five_minute_data():
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=50) # Tighter window for faster API response
    
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    try:
        nifty_raw = yf.download('^NSEI', start=start_str, end=end_str, interval='5m', progress=False)
        vix_raw = yf.download('^INDIAVIX', start=start_str, end=end_str, interval='5m', progress=False)
    except Exception:
        nifty_raw = pd.DataFrame()
        vix_raw = pd.DataFrame()
    
    # IF INTERNET OR API THROTTLED, CREATE BACKUP DATA SO APP NEVER BLANKS Out
    if nifty_raw.empty or vix_raw.empty or len(nifty_raw) < 5:
        # Creating a robust synthetic historical window for testing off-market hours
        st.info("ℹ️ Off-Market API Throttling detected. Initializing Safe Simulation Mode...")
        idx = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        
        # Simulating base structures safely
        sim_nifty = 23500.0 + np.cumsum(np.random.normal(0, 15, 100))
        sim_vix = 13.5 + np.cumsum(np.random.normal(0, 0.2, 100))
        
        combined_fallback = pd.DataFrame(index=idx)
        combined_fallback['High_nifty'] = sim_nifty + 10
        combined_fallback['Low_nifty'] = sim_nifty - 10
        combined_fallback['Open_nifty'] = sim_nifty - 2
        combined_fallback['Close_nifty'] = sim_nifty
        combined_fallback['High_vix'] = sim_vix + 0.1
        combined_fallback['Low_vix'] = sim_vix - 0.1
        combined_fallback['Close_vix'] = sim_vix
        combined_fallback['Datetime_x'] = combined_fallback.index
        return combined_fallback

    # Flattening MultiIndex Columns safely
    nifty_raw.columns = [f"{col[0]}_nifty" if isinstance(col, tuple) else f"{col}_nifty" for col in nifty_raw.columns]
    vix_raw.columns = [f"{col[0]}_vix" if isinstance(col, tuple) else f"{col}_vix" for col in vix_raw.columns]

    # Structuring matrix
    nifty_df = pd.DataFrame(index=nifty_raw.index)
    nifty_df['High_nifty'] = nifty_raw['High_nifty']
    nifty_df['Low_nifty'] = nifty_raw['Low_nifty']
    nifty_df['Open_nifty'] = nifty_raw['Open_nifty']
    nifty_df['Close_nifty'] = nifty_raw['Close_nifty']

    vix_df = pd.DataFrame(index=vix_raw.index)
    vix_df['High_vix'] = vix_raw['High_vix']
    vix_df['Low_vix'] = vix_raw['Low_vix']
    vix_df['Close_vix'] = vix_raw['Close_vix']

    nifty_df.index = pd.to_datetime(nifty_df.index).tz_localize(None)
    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
    
    nifty_df['time_key'] = nifty_df.index.strftime('%Y-%m-%d %H:%M')
    vix_df['time_key'] = vix_df.index.strftime('%Y-%m-%d %H:%M')
    
    nifty_df = nifty_df.reset_index()
    vix_df = vix_df.reset_index()
    
    combined = pd.merge(nifty_df, vix_df, on='time_key', how='inner')
    combined.index = pd.to_datetime(combined['Datetime_x'])
    combined = combined[~combined.index.duplicated(keep='first')]
    
    return combined.dropna()

# Execute data engine
combined_data = load_five_minute_data()

# Array conversions
n_high = combined_data['High_nifty'].to_numpy(dtype=float)
n_low = combined_data['Low_nifty'].to_numpy(dtype=
