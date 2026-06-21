import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty 50 2026 Live System", layout="wide")
st.title("🎯 Nifty 50 1-Hour Continuous Predictor (From Jan 2026)")
st.write("Tracking Nifty 50 (^NSEI) from January 1, 2026 onwards. Every historical candle evaluated live.")

# Fetch 1-Hour Accurate Nifty 50 Data from January 1, 2026
@st.cache_data(ttl=300)
def load_data():
    # Fetching data from January 2026 onwards for fresh market trend analysis
    df = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

df = load_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    # 2. Column A: (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Running Loop (The Balanced 0.03 Multiplier Zone)
    multiplier = 0.03 
    col_b = np.zeros(len(df))
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]
    for i in range(1, len(df)):
        col_b[i] = col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - col_b[i-1]))
    df['Column B'] = col_b
    
    # 4. Column C: A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 5. Continuous Nifty Volatility & Momentum Calculations
    df['Nifty_Range'] = df['High'] - df['Low']
    df['Avg_Nifty_Range'] = df['Nifty_Range'].rolling(window=20).mean()
    df['Nifty_Body'] = df['Close'] - df['Open']
    
    # 6. Column E: Continuous Live Behavioral Analysis on Nifty 50
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        curr_c = df['Column C'].iloc[i]
        n_range = df['Nifty_Range'].iloc[i]
        a_range = df['Avg_Nifty_Range'].iloc[i]
        body = df['Nifty_Body'].iloc[i]
        
        # Checking Range Expansion (Volatility Breakout)
        if pd.isna(a_range):
            is_range_expanded = False
        else:
            is_range_expanded = n_range > (a_range * 1.05)
        
        if curr_c > 0:  # Nifty Trend is Plus (+) -> Bullish Zone
            if body > 0 and is_range_expanded:
                status_list.append("🟢 BULLISH BREAKOUT (Strong Momentum)")
            elif body < 0:
                status_list.append("❌ FAKE UPMOVE (Price Dropping in Bull Trend Zone)")
            else:
                status_list.append("💤 SLOW ACCUMULATION (Weak Buying)")
        else:  # Nifty Trend is Minus (-) -> Bearish Zone
            if body < 0 and is_range_expanded:
                status_list.append("🔴 BEARISH BREAKDOWN (Strong Selling)")
            elif body > 0:
                status_list.append("❌ FAKE DOWNMOVE (Price Rising in Bear Trend Zone)")
            else:
                status_list.append("💤 SLOW DISTRIBUTION (Weak Momentum)")
                
    df['Column E'] = status_list
    
    # Keep data strictly from January 1, 2026 onwards
    df = df[df['Raw_Date'] >= '2026-01-01'].copy()
    
    # Final Grid Layout for Nifty 50 (Latest on Top)
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]
    
    def color_nifty_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;' # Green for Bullish
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;' # Red for Bearish
        if "❌" in val: return 'background-color: #e67e22; color: white; font-weight: bold;' # Orange for Trap
        if "💤" in val: return 'background-color: #7f8c8d; color: white;' # Grey for Sideways
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.2f}', 'Column C': '{:.2f}'
    }).map(color_nifty_grid, subset=['Column E']), use_container_width=True)
else:
    st.error("Nifty 50 historical data available nahi hai.")
