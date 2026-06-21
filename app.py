import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty Pro Volume ETF System", layout="wide")
st.title("🚀 Nifty 1-Hour Institutional Volume Matrix (Nifty BeES Synced)")
st.write("Formula: Col A=(H+L)/2 | Col B=0.0001 Loop | Col C=A-B | Col E=Nifty BeES Volume Filter")

# Fetch 1-Hour Nifty Price & Nifty BeES ETF Volume Data
@st.cache_data(ttl=300)
def load_synchronized_data():
    # 1. Fetch Nifty Index Price (Baseline frozen from 2026-01-01)
    df_index = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h")
    df_index.columns = [col[0] if isinstance(col, tuple) else col for col in df_index.columns]
    
    # 2. Fetch Nifty BeES ETF for Real Tradable Volume
    df_etf = yf.download(tickers="NIFTYBEES.NS", start="2026-01-01", interval="1h")
    df_etf.columns = [col[0] if isinstance(col, tuple) else col for col in df_etf.columns]
    
    if not df_index.empty:
        df_index = df_index.reset_index()
        
        if not df_etf.empty:
            df_etf = df_etf.reset_index()
            # Aligning ETF volumes based on exact datetime mapping
            index_time_col = 'Datetime' if 'Datetime' in df_index.columns else df_index.columns[0]
            etf_time_col = 'Datetime' if 'Datetime' in df_etf.columns else df_etf.columns[0]
            
            vol_map = dict(zip(df_etf[etf_time_col], df_etf['Volume']))
            df_index['True_Volume'] = df_index[index_time_col].map(vol_map).fillna(df_etf['Volume'].median())
        else:
            df_index['True_Volume'] = 100000  # Fallback base
            
        return df_index
    return pd.DataFrame()

df = load_synchronized_data()

if not df.empty:
    # 1. Column D: Date and Time
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Column D'] = pd.to_datetime(df
