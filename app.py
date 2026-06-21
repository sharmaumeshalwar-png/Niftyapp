import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="India VIX Pure Excel System 2025-2026", layout="wide")
st.title("🎯 India VIX 1-Hour Exact Excel Logic Predictor (From Jan 2025)")
st.write("Tracking India VIX (^INDIAVIX) from January 1, 2025 onwards with exact mathematical cell formulas.")

# Fetch 1-Hour Accurate India VIX Data from January 1, 2025
@st.cache_data(ttl=300)
def load_data():
    df = yf.download(tickers="^INDIAVIX", start="2025-01-01", interval="1h")
    # Flatten multi-index columns if present in yfinance output
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

df = load_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time Formatter
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    # 2. Column A: Exact Formula -> (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Exact Excel Drag-Down Loop Logic
    # 1st Row: =A1
    # 2nd Row: =B1 + 0.0001 * (A2 - B1)
    multiplier = 0.0001
    col_b = np.zeros(len(df))
    
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]  # First row logic (=A1)
        
    for i in range(1, len(df)):
        # Linear row-by-row tracing mimicking the Excel cell dependencies
        col_b[i] = col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - col_b[i-1]))
        
    df['Column B'] = col_b
    
    # 4. Column C: Exact Formula -> A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 5. Volatility Indicators for Behavioral Pattern Tracking
    df['VIX_Range'] = df['High'] - df['Low']
    df['Avg_VIX_Range'] = df['VIX_Range'].rolling(window=20).mean()
    df['VIX_Body'] = df['Close'] - df['Open']
    
    # 6. Column E: Behavioral Analysis on Top of the Formulas (VIX Color Scheme)
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        curr_c = df['Column C'].iloc[i]
        v_range = df['VIX_Range'].iloc[i]
        a_range = df['Avg_VIX_Range'].iloc[i]
        body = df['VIX_Body'].iloc[i]
        
        if pd.isna(a_range):
            is_range_expanded = False
        else:
            is_range_expanded = v_range > (a_range * 1.05)
        
        # Grid classification engine based on Column C polarity (VIX-optimized)
        if curr_c > 0:  # Column C is Plus (+) -> Fear/Volatility Spike Zone
            if body > 0 and is_range_expanded:
                status_list.append("🟢 FEAR SPIKE RUNNING (Strong VIX Expansion)")
            elif body < 0:
                status_list.append("❌ FAKE VIX RISE (Price Dropping in VIX Bull Trend)")
            else:
                status_list.append("💤 SLOW FEAR BUILDING (Weak VIX Momentum)")
        else:  # Column C is Minus (-) -> Market Cooling/Calming Zone
            if body < 0 and is_range_expanded:
                status_list.append("🔴 VIX COOLING RUNNING (Strong Drop Continuation)")
            elif body > 0:
                status_list.append("❌ FAKE VIX DROP (Price Rising in VIX Bear Trend)")
            else:
                status_list.append("💤 SLOW CALMING (Weak Down Momentum)")
                
    df['Column E'] = status_list
    
    # Strictly filter from January 1, 2025 onwards
    df = df[df['Raw_Date'] >= '2025-01-01'].copy()
    
    # Reverse final layout to
