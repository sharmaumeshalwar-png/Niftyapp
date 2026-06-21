import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty Pure Price System", layout="wide")
st.title("🎯 Nifty 1-Hour Pure Price Action Dashboard")
st.write("Fixed: Volume completely removed. Column E filters traps using Pure Price Range Expansion & Velocity.")

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
    
    # 3. Column B: Running Loop (Excel Logic - 0.0001 multiplier ONLY here)
    multiplier = 0.0001
    col_b = np.zeros(len(df))
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]
    for i in range(1, len(df)):
        col_b[i] = col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - col_b[i-1]))
    df['Column B'] = col_b
    
    # 4. Column C: A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 5. Pure Price Action Calculations (Independent of Volume & Multiplier)
    df['Candle_Range'] = df['High'] - df['Low']
    df['Avg_Range_20'] = df['Candle_Range'].rolling(window=20).mean()
    df['Real_Body'] = df['Close'] - df['Open']
    
    # 6. Column E: Conditional Labeling ONLY on Column C Sign Changes
    status_list = [""] # Baseline placeholder
    
    for i in range(1, len(df)):
        prev_c = df['Column C'].iloc[i-1]
        curr_c = df['Column C'].iloc[i]
        c_range = df['Candle_Range'].iloc[i]
        a_range = df['Avg_Range_20'].iloc[i]
        body = df['Real_Body'].iloc[i]
        
        # Identify Sign Flip (+ to - OR - to +)
        sign_changed = (prev_c >= 0 and curr_c < 0) or (prev_c < 0 and curr_c >= 0)
        
        if sign_changed:
            # Price Action Rule: Is the current candle range expanding significantly?
            is_range_expanded = c_range > (a_range * 1.1)
            
            if curr_c > 0:  # Sign flipped to Plus (+)
                if is_range_expanded and body > 0:
                    status_list.append("🟢 VALID BREAKOUT (Strong Price Push)")
                else:
                    status_list.append("❌ FAKE BREAKOUT (Weak Price Trap)")
            else:  # Sign flipped to Minus (-)
                if is_range_expanded and body < 0:
                    status_list.append("🔴 VALID BREAKDOWN (Strong Price Drop)")
                else:
                    status_list.append("❌ FAKE BREAKDOWN (Weak Price Trap)")
        else:
            # Trend is continuing normally, keep cell 100% blank
            status_list.append("")
            
    df['Column E'] = status_list
    
    # Complete Layout Data Grid (All rows preserved)
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]  # Latest candle on top
    
    def color_pure_price(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "❌" in val: return 'background-color: #e67e22; color: white; font-weight: bold;'
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}'
    }).map(color_pure_price, subset=['Column E']), use_container_width=True)
else:
    st.error("Market data load nahi ho pa raha hai.")
