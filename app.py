import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty March 27 Targeted System", layout="wide")
st.title("🎯 Nifty 1-Hour Continuous Predictor (From March 27 onwards)")
st.write("Column E: Active on EVERY single candle, capturing dynamic momentum changes and immediate traps.")

# Fetch 1-Hour Accurate Data starting strictly from March 27, 2026
@st.cache_data(ttl=300)
def load_data():
    # Fetching a bit earlier to let rolling 20-period averages settle perfectly by March 27
    df = yf.download(tickers="^NSEI", start="2026-03-15", interval="1h")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

df = load_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %H:%M')
    
    # 2. Column A: (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Running Loop (Excel Formula Multiplier - 0.0001)
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
    
    # 6. Column E: Continuous Live Behavioral Analysis
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        curr_c = df['Column C'].iloc[i]
        c_range = df['Candle_Range'].iloc[i]
        a_range = df['Avg_Range_20'].iloc[i]
        body = df['Real_Body'].iloc[i]
        
        is_range_expanded = c_range > (a_range * 1.05)
        
        if curr_c > 0:  # Trend is Plus (+)
            if body > 0 and is_range_expanded:
                status_list.append("🟢 BULLISH RUNNING (Strong Continuation)")
            elif body < 0:
                status_list.append("❌ FAKE MOVE TRAP (Price Dropping in Bull Trend)")
            else:
                status_list.append("💤 WEAK MOMENTUM (Slow Buying Zone)")
        else:  # Trend is Minus (-)
            if body < 0 and is_range_expanded:
                status_list.append("🔴 BEARISH RUNNING (Strong Drop Continuation)")
            elif body > 0:
                status_list.append("❌ FAKE MOVE TRAP (Price Rising in Bear Trend)")
            else:
                status_list.append("💤 WEAK MOMENTUM (Slow Selling Zone)")
                
    df['Column E'] = status_list
    
    # Strict Filter: Keep data only from March 27, 2026 onwards
    df = df[df['Raw_Date'] >= '2026-03-27'].copy()
    
    # Final Grid Layout (Latest on Top)
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]
    
    def color_dynamic_grid(val):
        if "🟢" in val or "🔴" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "❌" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "💤" in val: return 'background-color: #e67e22; color: white; font-weight: bold;'
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}'
    }).map(color_dynamic_grid, subset=['Column E']), use_container_width=True)
else:
    st.error("Data processing failed.")
