import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty-VIX 90% Trap Filter June 2026", layout="wide")
st.title("🎯 Nifty 50 + India VIX 1-Hour Traps & Divergence System (June 2026)")
st.write("Tracking Nifty (^NSEI) and VIX (^INDIAVIX) from June 1, 2026 onwards. Filtering the 90% Fake Out Moves.")

# Fetch 1-Hour Data safely by downloading separately to bypass yfinance multi-ticker hourly limits
@st.cache_data(ttl=300)
def load_combined_data():
    # Setting the exact baseline date requested: June 1, 2026
    df_nifty = yf.download(tickers="^NSEI", start="2026-06-01", interval="1h")
    df_vix = yf.download(tickers="^INDIAVIX", start="2026-06-01", interval="1h")
    
    # Flatten columns if multi-index is present
    df_nifty.columns = [col[0] if isinstance(col, tuple) else col for col in df_nifty.columns]
    df_vix.columns = [col[0] if isinstance(col, tuple) else col for col in df_vix.columns]
    
    if df_nifty.empty or df_vix.empty:
        return pd.DataFrame()
        
    # Rename columns to distinguish between Nifty and VIX before merging
    df_nifty = df_nifty[['Open', 'High', 'Low', 'Close']].rename(
        columns={'Open': 'N_Open', 'High': 'N_High', 'Low': 'N_Low', 'Close': 'N_Close'}
    )
    df_vix = df_vix[['Open', 'High', 'Low', 'Close']].rename(
        columns={'Open': 'V_Open', 'High': 'V_High', 'Low': 'V_Low', 'Close': 'V_Close'}
    )
    
    # Merge on the exact Datetime index to sync both datasets row-by-row
    combined_df = pd.merge(df_nifty, df_vix, left_index=True, right_index=True, how='inner')
    return combined_df

df = load_combined_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time Formatter
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    # 2. Column A Calculations: Exact Formula -> (High + Low) / 2
    df['Nifty_A'] = (df['N_High'] + df['N_Low']) / 2
    df['Vix_A'] = (df['V_High'] + df['V_Low']) / 2
    
    # 3. Column B Calculations: Exact Excel Drag-Down Loop Logic (Multiplier = 0.0001)
    multiplier = 0.0001
    n_col_b = np.zeros(len(df))
    v_col_b = np.zeros(len(df))
    
    if len(df) > 0:
        n_col_b[0] = df['Nifty_A'].iloc[0]  # First row Nifty = A1 logic
        v_col_b[0] = df['Vix_A'].iloc[0]    # First row VIX = A1 logic
        
    for i in range(1, len(df)):
        # Replicating row-by-row sequence dependency
        n_col_b[i] = n_col_b[i-1] + (multiplier * (df['Nifty_A'].
