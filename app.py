import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty Aligned Volume System", layout="wide")
st.title("🎯 Nifty 1-Hour Price Matrix & Candle-Volume Alignment")
st.write("Framework: Full Data Preserved. Column E aligns Volume + Candle Size ONLY on Column C Sign Changes.")

# Fetch 1-Hour Synchronized Data (Nifty Index Price + NiftyBees Volume)
@st.cache_data(ttl=300)
def load_data():
    df_idx = yf.download(tickers="^NSEI", start="2026-01-01", interval="1h")
    df_idx.columns = [col[0] if isinstance(col, tuple) else col for col in df_idx.columns]
    
    df_bees = yf.download(tickers="NIFTYBEES.NS", start="2026-01-01", interval="1h")
    df_bees.columns = [col[0] if isinstance(col, tuple) else col for col in df_bees.columns]
    
    if not df_idx.empty:
        df_idx = df_idx.reset_index()
        if not df_bees.empty:
            df_bees = df_bees.reset_index()
            t_idx = 'Datetime' if 'Datetime' in df_idx.columns else df_idx.columns[0]
            t_bees = 'Datetime' if 'Datetime' in df_bees.columns else df_bees.columns[0]
            bees_vol_map = dict(zip(df_bees[t_bees], df_bees['Volume']))
            df_idx['True_Volume'] = df_idx[t_idx].map(bees_vol_map).fillna(df_bees['Volume'].median())
        else:
            df_idx['True_Volume'] = 500000
        return df_idx
    return pd.DataFrame()

df = load_data()

if not df.empty:
    # 1. Column D: Date and Time
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Column D'] = pd.to_datetime(df[time_col]).dt.strftime('%d %b %H:%M')
    
    # 2. Column A: (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2
    
    # 3. Column B: Pure Price Loop (Excel Formula Logic - 0.0001 multiplier ONLY here)
    multiplier = 0.0001
    col_b = np.zeros(len(df))
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]
    for i in range(1, len(df)):
        col_b[i] = col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - col_b[i-1]))
    df['Column B'] = col_b
    
    # 4. Column C: A - B (Sign Flip Base)
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 5. Volume and Candle Size Metrics (No Excel Multiplier Connection)
    df['Vol_Avg'] = df['True_Volume'].rolling(window=20).mean()
    df['Vol_Std'] = df['True_Volume'].rolling(window=20).std()
    df['Candle_Body'] = abs(df['Close'] - df['Open'])
    df['Avg_Body'] = df['Candle_Body'].rolling(window=20).mean()
    
    # 6. Column E: Precise Candle-Volume Alignment on Sign Flip Only
    status_list = [""] # Base row placeholder
    
    for i in range(1, len(df)):
        prev_c = df['Column C'].iloc[i-1]
        curr_c = df['Column C'].iloc[i]
        vol = df['True_Volume'].iloc[i]
        v_avg = df['Vol_Avg'].iloc[i]
        v_std = df['Vol_Std'].iloc[i]
        body = df['Candle_Body'].iloc[i]
        avg_body = df['Avg_Body'].iloc[i]
        
        # Check Sign Flip
        sign_changed = (prev_c >= 0 and curr_c < 0) or (prev_c < 0 and curr_c >= 0)
        
        if sign_changed:
            # Check if volume is heavy (Greater than past average)
            is_high_vol = vol > (v_avg + 0.3 * v_std)
            # Check if candle size is large or small
            is_large_candle = body > avg_body
            
            if curr_c > 0:  # Sign flipped to Plus (+)
                if is_high_vol and is_large_candle:
                    status_list.append("🟢 ASLI BUY HINT (Volume + Candle Aligned)")
                elif is_high_vol and not is_large_candle:
                    status_list.append("🐳 ACCUMULATION (Absorption Block)")
                elif not is_high_vol and is_large_candle:
                    status_list.append("❌ NO-VOLUME TRAP (Fake Buy)")
                else:
                    status_list.append("💤 RETAIL FLIP (Weak Momentum)")
            else:  # Sign flipped to Minus (-)
                if is_high_vol and is_large_candle:
                    status_list.append("🔴 ASLI SELL HINT (Volume + Candle Aligned)")
                elif is_high_vol and not is_large_candle:
                    status_list.append("🐻 DISTRIBUTION (Supply Pressure)")
                elif not is_high_vol and is_large_candle:
                    status_list.append("❌ NO-VOLUME TRAP (Fake Sell)")
                else:
                    status_list.append("💤 RETAIL FLIP (Weak Momentum)")
        else:
            # No sign change -> Keep Column E 100% empty
            status_list.append("")
            
    df['Column E'] = status_list
    
    # Complete Grid Display (All candles preserved)
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]  # Latest on top
    
    def color_vsa_matrix(val):
        if "🟢" in val or "🐳" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val or "🐻" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "❌" in val: return 'background-color: #e67e22; color: white; font-weight: bold;'
        return ''

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}'
    }).map(color_vsa_matrix, subset=['Column E']), use_container_width=True)
else:
    st.error("Data synchronization error.")
