import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty Pro Volume System", layout="wide")
st.title("🚀 Nifty 1-Hour Institutional Volume Matrix (NiftyBees Synced)")
st.write("Formula: Col A=(H+L)/2 | Col B=0.0001 Loop | Col C=A-B | Col E=NiftyBees Pro Volume Filter")

# Fetch 1-Hour Data Safely
@st.cache_data(ttl=300)
def load_data():
    # 1. Fetch Nifty Index for accurate Price Loop
    df_idx = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h")
    df_idx.columns = [col[0] if isinstance(col, tuple) else col for col in df_idx.columns]
    
    # 2. Fetch NiftyBees ETF for Real Deployed Volume
    df_bees = yf.download(tickers="NIFTYBEES.NS", start="2026-01-01", interval="1h")
    df_bees.columns = [col[0] if isinstance(col, tuple) else col for col in df_bees.columns]
    
    if not df_idx.empty:
        df_idx = df_idx.reset_index()
        if not df_bees.empty:
            df_bees = df_bees.reset_index()
            
            # Extract time column name dynamically
            t_idx = 'Datetime' if 'Datetime' in df_idx.columns else df_idx.columns[0]
            t_bees = 'Datetime' if 'Datetime' in df_bees.columns else df_bees.columns[0]
            
            # Map NiftyBees Volume to Index Datetime
            bees_vol_map = dict(zip(df_bees[t_bees], df_bees['Volume']))
            df_idx['True_Volume'] = df_idx[t_idx].map(bees_vol_map).fillna(df_bees['Volume'].median())
        else:
            df_idx['True_Volume'] = 500000  # Fallback base
        return df_idx
    return pd.DataFrame()

df = load_data()

if not df.empty:
    # 1. Column D: Date and Time formatting
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Column D'] = pd.to_datetime(df[time_col]).dt.strftime('%d %b %H:%M')
    
    # 2. Column A: (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Running Loop with 0.0001 multiplier
    multiplier = 0.0001
    col_b = np.zeros(len(df))
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]
        
    for i in range(1, len(df)):
        a_current = df['Column A'].iloc[i]
        b_prev = col_b[i-1]
        col_b[i] = b_prev + (multiplier * (a_current - b_prev))
    df['Column B'] = col_b
    
    # 4. Column C: A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 5. Column E: Institutional Pro Volume Filter Logic
    df['Vol_Mean'] = df['True_Volume'].rolling(window=20).mean()
    df['Vol_Std'] = df['True_Volume'].rolling(window=20).std()
    
    status_list = []
    for i in range(len(df)):
        if i < 20:
            status_list.append("Institutional Loading...")
            continue
            
        vol = df['True_Volume'].iloc[i]
        v_mean = df['Vol_Mean'].iloc[i]
        v_std = df['Vol_Std'].iloc[i]
        c_val = df['Column C'].iloc[i]
        close_p = df['Close'].iloc[i]
        open_p = df['Open'].iloc[i]
        
        # Proper Z-Score Volume Thresholding
        is_heavy_volume = vol > v_mean
        is_extreme_volume = vol > (v_mean + 0.8 * v_std)
        is_green_candle = close_p > open_p
        
        if is_extreme_volume:
            if c_val > 0 and is_green_candle:
                status_list.append("🔥 INSTITUTIONAL BLAST (Buy Confirm)")
            elif c_val < 0 and not is_green_candle:
                status_list.append("⚠️ INSTITUTIONAL DUMP (Sell Confirm)")
            elif c_val < 0 and is_green_candle:
                status_list.append("🐳 ACCUMULATION TRAP (Buy Hint)")
            else:
                status_list.append("🐻 DISTRIBUTION TRAP (Sell Hint)")
        elif is_heavy_volume:
            if c_val > 0:
                status_list.append("✅ TREND CONFIRMED (Buy)")
            else:
                status_list.append("🚨 TREND CONFIRMED (Sell)")
        else:
            if abs(c_val) > 10:
                status_list.append("❌ FAKE MOVE (No Volume)")
            else:
                status_list.append("💤 SIDEWAYS (Retail Churn)")
                
    df['Column E'] = status_list
    
    # Top Metrics
    latest_a = float(df['Column A'].iloc[-1])
    latest_b = float(df['Column B'].iloc[-1])
    latest_c = float(df['Column C'].iloc[-1])
    latest_e = str(df['Column E'].iloc[-1])
    latest_time = str(df['Column D'].iloc[-1])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Candle Time (Col D)", latest_time)
    col2.metric("Column C (A - B)", f"{latest_c:.4f}")
    col3.metric("Smart Filter (Col E)", latest_e)
    
    st.subheader("📋 Pro Institutional Data Grid (Last 20 Candles)")
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]  # Latest on top
    
    def color_status(val):
        if "🔥" in val or "✅" in val: return 'background-color: #1fc07c; color: white'
        if "⚠️" in val or "🚨" in val: return 'background-color: #ff4b4b; color: white'
        if "🐳" in val or "🐻" in val: return 'background-color: #0072b2; color: white'
        return 'background-color: #2b2b2b; color: #888888'

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 
        'Column B': '{:.4f}', 
        'Column C': '{:.4f}'
    }).map(color_status, subset=['Column E']), use_container_width=True)
else:
    st.error("Market data load nahi ho pa raha hai.")
