import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="India VIX 2025-2026 System", layout="wide")
st.title("🎯 India VIX 1-Hour Continuous Predictor (From June 2025)")
st.write("Tracking India VIX (^INDIAVIX) from June 1, 2025 onwards. Every historical candle evaluated live.")

# Fetch 1-Hour Accurate India VIX Data from June 1, 2025
@st.cache_data(ttl=300)
def load_data():
    # Fetching directly from June 2025 for long-term historical analysis
    df = yf.download(tickers="^INDIAVIX", start="2025-06-01", interval="1h")
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
    
    # 3. Column B: Running Loop (Excel Logic - 0.0001 multiplier applied to VIX)
    multiplier = 0.0001
    col_b = np.zeros(len(df))
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]
    for i in range(1, len(df)):
        col_b[i] = col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - col_b[i-1]))
    df['Column B'] = col_b
    
    # 4. Column C: A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 5. Continuous VIX Volatility Calculations
    df['VIX_Range'] = df['High'] - df['Low']
    df['Avg_VIX_Range'] = df['VIX_Range'].rolling(window=20).mean()
    df['VIX_Body'] = df['Close'] - df['Open']
    
    # 6. Column E: Continuous Live Behavioral Analysis on India VIX
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        curr_c = df['Column C'].iloc[i]
        v_range = df['VIX_Range'].iloc[i]
        a_range = df['Avg_VIX_Range'].iloc[i]
        body = df['VIX_Body'].iloc[i]
        
        # Checking VIX Range Expansion
        is_range_expanded = v_range > (a_range * 1.05)
        
        if curr_c > 0:  # VIX Trend is Plus (+) -> Fear Spike
            if body > 0 and is_range_expanded:
                status_list.append("🟢 FEAR SPIKE RUNNING (Strong VIX Expansion)")
            elif body < 0:
                status_list.append("❌ FAKE VIX RISE (Price Dropping in VIX Bull Trend)")
            else:
                status_list.append("💤 SLOW FEAR BUILDING (Weak VIX Momentum)")
        else:  # VIX Trend is Minus (-) -> Market Cooling
            if body < 0 and is_range_expanded:
                status_list.append("🔴 VIX COOLING RUNNING (Strong Drop Continuation)")
            elif body > 0:
                status_list.append("❌ FAKE VIX DROP (Price Rising in VIX Bear Trend)")
            else:
                status_list.append("💤 SLOW CALMING (Weak Down Momentum)")
                
    df['Column E'] = status_list
    
    # Keep data strictly from June 1, 2025 onwards
    df = df[df['Raw_Date'] >= '2025-06-01'].copy()
    
    # Final Grid Layout for India VIX (Latest on Top)
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]
    
    def color_vix_grid(val):
        if "🟢" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;' # Red for Fear Spike
        if "🔴" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;' # Green for Calm Market
        if "❌" in val: return 'background-color: #e67e22; color: white; font-weight: bold;' # Orange for Trap
        if "💤" in val: return 'background-color: #7f8c8d; color: white;' # Grey for Sideways
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}'
    }).map(color_vix_grid, subset=['Column E']), use_container_width=True)
else:
    st.error("India VIX long-term data available nahi hai.")
