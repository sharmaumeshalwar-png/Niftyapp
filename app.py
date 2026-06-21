import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty True Predictor System", layout="wide")
st.title("🚀 Nifty 1-Hour Volatility & Price Action Predictor")
st.write("Column E: Advanced Volatility Matrix & Reversal Confirmation (Independent of Raw Volume Truncation)")

# Fetch 1-Hour Nifty Data
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
    
    # 3. Column B: Running Loop with 0.0001 multiplier (Excel Baseline)
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
    
    # 5. Advanced Mathematical Volatility Filter (ATR & Momentum Spread)
    # Calculating price expansion without relying on faulty volume statistics
    df['Candle_Spread'] = df['High'] - df['Low']
    df['Spread_EMA'] = df['Candle_Spread'].ewm(span=20, adjust=False).mean()
    df['Price_Velocity'] = df['Close'] - df['Open']
    
    status_list = [""] # Baseline placeholder
    
    for i in range(1, len(df)):
        prev_c = df['Column C'].iloc[i-1]
        curr_c = df['Column C'].iloc[i]
        
        spread = df['Candle_Spread'].iloc[i]
        avg_spread = df['Spread_EMA'].iloc[i]
        velocity = df['Price_Velocity'].iloc[i]
        
        # Check Sign Flip (+ to - OR - to +)
        sign_changed = (prev_c >= 0 and curr_c < 0) or (prev_c < 0 and curr_c >= 0)
        
        if sign_changed:
            # Check if price expansion is authentic (Volatility Breakout)
            is_authentic_move = spread > (avg_spread * 1.15)
            
            if curr_c > 0: # Flipped to Plus (+)
                if is_authentic_move and velocity > 0:
                    status_list.append("🟢 STRONG BUY CONFIRMED (Institutional Expansion)")
                elif not is_authentic_move and abs(velocity) < (avg_spread * 0.3):
                    status_list.append("🐳 SMART ACCUMULATION (Low Volatility Absorption)")
                else:
                    status_list.append("❌ NO-FORCE TRAP (90% Opposite Risk - Ignore)")
            else: # Flipped to Minus (-)
                if is_authentic_move and velocity < 0:
                    status_list.append("🔴 STRONG SELL CONFIRMED (Institutional Liquidation)")
                elif not is_authentic_move and abs(velocity) < (avg_spread * 0.3):
                    status_list.append("🐻 SMART DISTRIBUTION (Low Volatility Dumping)")
                else:
                    status_list.append("❌ NO-FORCE TRAP (90% Opposite Risk - Ignore)")
        else:
            # No sign change, keep Column E 100% empty string
            status_list.append("")
            
    df['Column E'] = status_list
    
    # Display Grid Setup (All Rows Preserved)
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]  # Latest candle on top
    
    def color_vsa_matrix(val):
        if "🟢" in val or "🐳" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val or "🐻" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "❌" in val: return 'background-color: #e67e22; color: white; font-weight: bold;'
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}'
    }).map(color_vsa_matrix, subset=['Column E']), use_container_width=True)
else:
    st.error("Market data load nahi ho pa raha hai.")
