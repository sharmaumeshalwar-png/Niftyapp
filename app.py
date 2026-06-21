import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty Live Candle Tracker", layout="wide")
st.title("🎯 Nifty 1-Hour Continuous Price Action Predictor")
st.write("Column E: Analyzes EVERY candle live based on current momentum and range expansions (No Sign-Change Waiting).")

# Fetch 1-Hour Accurate Nifty Index Data
@st.cache_data(ttl=300)
def load_data():
    df = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

df = load_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Column D'] = pd.to_datetime(df[time_col]).dt.strftime('%d %b %H:%M')
    
    # 2. Column A: (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Running Loop (Excel Logic - 0.0001 multiplier)
    multiplier = 0.0001
    col_b = np.zeros(len(df))
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]
    for i in range(1, len(df)):
        col_b[i] = col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - col_b[i-1]))
    df['Column B'] = col_b
    
    # 4. Column C: A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 5. Continuous Price Action Math
    df['Candle_Range'] = df['High'] - df['Low']
    df['Avg_Range_20'] = df['Candle_Range'].rolling(window=20).mean()
    df['Real_Body'] = df['Close'] - df['Open']
    
    # 6. Column E: Active Evaluation on EVERY Single Candle
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        curr_c = df['Column C'].iloc[i]
        c_range = df['Candle_Range'].iloc[i]
        a_range = df['Avg_Range_20'].iloc[i]
        body = df['Real_Body'].iloc[i]
        
        # Checking range expansion for current candle strength
        is_range_expanded = c_range > (a_range * 1.05)
        
        if curr_c > 0: # Current Trend is Plus (+)
            if body > 0 and is_range_expanded:
                status_list.append("🟢 BULLISH RUNNING (Strong Continuation)")
            elif body < 0:
                status_list.append("❌ FAKE MOVE TRAP (Price Dropping in Bull Trend)")
            else:
                status_list.append("💤 WEAK MOMENTUM (Slow Buying Zone)")
                
        else: # Current Trend is Minus (-)
            if body < 0 and is_range_expanded:
                status_list.append("🔴 BEARISH RUNNING (Strong Drop Continuation)")
            elif body > 0:
                status_list.append("❌ FAKE MOVE TRAP (Price Rising in Bear Trend)")
            else:
                status_list.append("💤 WEAK MOMENTUM (Slow Selling Zone)")
                
    df['Column E'] = status_list
    
    # Full Grid Layout
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]  # Latest on top
    
    def color_every_candle(val):
        if "🟢" in val or "🔴" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "❌" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "💤" in val: return 'background-color: #e67e22; color: white; font-weight: bold;'
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}'
    }).map(color_every_candle, subset=['Column E']), use_container_width=True)
else:
    st.error("Data load nahi ho pa raha hai.")
