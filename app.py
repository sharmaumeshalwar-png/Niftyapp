import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty Volume Behavior Tracker", layout="wide")
st.title("🎯 Nifty 1-Hour Pure Volume Spread (VSA) Tracker")
st.write("System Focus: Displays market behavior ONLY when Column C changes its Sign (+ / -)")

# Fetch 1-Hour Synchronized Data (Nifty Index + NiftyBees Volume)
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
    
    # 3. Column B: Running Loop (0.0001 multiplier)
    multiplier = 0.0001
    col_b = np.zeros(len(df))
    if len(df) > 0:
        col_b[0] = df['Column A'].iloc[0]
    for i in range(1, len(df)):
        col_b[i] = col_b[i-1] + (multiplier * (df['Column A'].iloc[i] - col_b[i-1]))
    df['Column B'] = col_b
    
    # 4. Column C: A - B
    df['Column C'] = df['Column A'] - df['Column B']
    
    # 5. Volume Analysis Baseline (20-period Mean and Std Dev)
    df['Vol_Avg'] = df['True_Volume'].rolling(window=20).mean()
    df['Vol_Std'] = df['True_Volume'].rolling(window=20).std()
    
    # 6. Column E: VSA Behavior Logic on Sign Flip
    status_list = ["Baseline"]
    
    for i in range(1, len(df)):
        prev_c = df['Column C'].iloc[i-1]
        curr_c = df['Column C'].iloc[i]
        vol = df['True_Volume'].iloc[i]
        v_avg = df['Vol_Avg'].iloc[i]
        v_std = df['Vol_Std'].iloc[i]
        
        # Determine if candle spread is large or small
        candle_body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        avg_candle_body = abs(df['Close'] - df['Open']).rolling(window=20).mean().iloc[i]
        
        # Check Sign Flip
        sign_changed = (prev_c >= 0 and curr_c < 0) or (prev_c < 0 and curr_c >= 0)
        
        if sign_changed:
            is_high_vol = vol > (v_avg + 0.5 * v_std)
            is_large_candle = candle_body > avg_candle_body
            
            if curr_c > 0:  # Sign flipped to Plus (+)
                if is_high_vol and is_large_candle:
                    status_list.append("🟢 ASLI BUY HINT (Whale Entry)")
                elif is_high_vol and not is_large_candle:
                    status_list.append("🐳 ACCUMULATION (Heavy Buying Absorption)")
                elif not is_high_vol and is_large_candle:
                    status_list.append("❌ NO-VOLUME TRAP (Fake Bull Move)")
                else:
                    status_list.append("💤 RETAIL FLIP (Weak Buy Signal)")
            else:  # Sign flipped to Minus (-)
                if is_high_vol and is_large_candle:
                    status_list.append("🔴 ASLI SELL HINT (Whale Dumping)")
                elif is_high_vol and not is_large_candle:
                    status_list.append("🐻 DISTRIBUTION (Heavy Selling Pressure)")
                elif not is_high_vol and is_large_candle:
                    status_list.append("❌ NO-VOLUME TRAP (Fake Bear Move)")
                else:
                    status_list.append("💤 RETAIL FLIP (Weak Sell Signal)")
        else:
            # No sign change, keep trend quiet
            status_list.append("Trend Continuing...")
            
    df['Column E'] = status_list
    
    # Filter only the rows where sign changed to analyze easily
    # We always include the very last row so the user sees the latest live candle behavior
    df['Sign_Changed'] = (df['Column C'].shift(1) >= 0) != (df['Column C'] >= 0)
    df.loc[df.index[-1], 'Sign_Changed'] = True  # Keep latest live candle visible
    
    filtered_df = df[df['Sign_Changed'] == True].copy()
    show_df = filtered_df[['Column D', 'Column A', 'Column B', 'Column C', 'Column E']].copy()
    show_df = show_df.iloc[::-1]  # Latest on top
    
    def color_vsa(val):
        if "🟢" in val or "🐳" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val or "🐻" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "❌" in val: return 'background-color: #e67e22; color: white; font-weight: bold;'
        return 'color: #888888;'

    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}'
    }).map(color_vsa, subset=['Column E']), use_container_width=True)
else:
    st.error("Data fetch error.")
