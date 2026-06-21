import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty-VIX 90% Trap Filter May 2026", layout="wide")
st.title("🎯 Nifty 50 + India VIX 1-Hour Traps & Divergence System (May 2026)")
st.write("Tracking Nifty (^NSEI) and VIX (^INDIAVIX) from May 1, 2026 onwards. Filtering the 90% Fake Out Moves.")

# Fetch 1-Hour Data safely by downloading separately to bypass yfinance multi-ticker hourly limits
@st.cache_data(ttl=300)
def load_combined_data():
    # Setting the exact baseline date requested: May 1, 2026
    df_nifty = yf.download(tickers="^NSEI", start="2026-05-01", interval="1h")
    df_vix = yf.download(tickers="^INDIAVIX", start="2026-05-01", interval="1h")
    
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
        n_col_b[i] = n_col_b[i-1] + (multiplier * (df['Nifty_A'].iloc[i] - n_col_b[i-1]))
        v_col_b[i] = v_col_b[i-1] + (multiplier * (df['Vix_A'].iloc[i] - v_col_b[i-1]))
        
    df['Nifty_B'] = n_col_b
    df['Vix_B'] = v_col_b
    
    # 4. Column C Calculations: Exact Formula -> A - B Deviation
    df['Nifty_C'] = df['Nifty_A'] - df['Nifty_B']
    df['Vix_C'] = df['Vix_A'] - df['Vix_B']
    
    # Range Metrics for Breakout Confirmations
    df['Nifty_Range'] = df['N_High'] - df['N_Low']
    df['Avg_Nifty_Range'] = df['Nifty_Range'].rolling(window=20).mean()
    df['Nifty_Body'] = df['N_Close'] - df['N_Open']
    
    # 5. Column E: The 90% Opposite Intersecting Trap Engine
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        n_curr_c = df['Nifty_C'].iloc[i]
        v_curr_c = df['Vix_C'].iloc[i]
        n_range = df['Nifty_Range'].iloc[i]
        a_range = df['Avg_Nifty_Range'].iloc[i]
        body = df['Nifty_Body'].iloc[i]
        
        if pd.isna(a_range):
            is_range_expanded = False
        else:
            is_range_expanded = n_range > (a_range * 1.05)
            
        # --- EXECUTION OF THE TRAP FILTER MATRIX ---
        if n_curr_c > 0:  # Nifty is in Plus (+) Trend Zone
            if v_curr_c <= 0:  # VIX is properly dropping in Minus (-) Zone -> 10% Real Moves
                if body > 0 and is_range_expanded:
                    status_list.append("🟢 TRUE BULLISH MOMENTUM (VIX Confirmed)")
                else:
                    status_list.append("💤 SLOW ACCUMULATION (Safe Buying Zone)")
            else:  # VIX is ALSO Tracking Plus (+) -> THE 90% TRAP DETECTED!
                status_list.append("⚠️ 90% FAKE UPMOVE TRAP (VIX Rising, Danger!)")
                
        else:  # Nifty is in Minus (-) Trend Zone
            if v_curr_c > 0:  # VIX is properly spiking in Plus (+) Zone -> Genuine Panic
                if body < 0 and is_range_expanded:
                    status_list.append("🔴 TRUE BEARISH CRASH (VIX Confirmed)")
                else:
                    status_list.append("💤 SLOW DISTRIBUTION (Safe Selling Zone)")
            else:  # VIX is ALSO Tracking Minus (-) -> THE 90% TRAP DETECTED!
                status_list.append("⚠️ 90% FAKE DOWNMOVE TRAP (VIX Dropping, Reversal Coming!)")
                
    df['Column E'] = status_list
    
    # Filter view to strictly show from May 1, 2026 onwards
    df = df[df['Raw_Date'] >= '2026-05-01'].copy()
    
    # Reverse final matrix data view to push latest candles to the top row
    show_df = df[['Column D', 'Nifty_A', 'Nifty_B', 'Nifty_C', 'Vix_C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]
    
    # Grid Styling Setup
    def color_trap_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
        if "💤" in val: return 'background-color: #34495e; color: #bdc3c7;'
        return ''

    st.dataframe(show_df.style.format({
        'Nifty_A': '{:.2f}', 'Nifty_B': '{:.4f}', 'Nifty_C': '{:.4f}', 'Vix_C': '{:.4f}'
    }).map(color_trap_grid, subset=['Column E']), use_container_width=True)
else:
    st.error("Data fetch error: Yahoo Finance se May 2026 ka hourly data load nahi ho paya.")
