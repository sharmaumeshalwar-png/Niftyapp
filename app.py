import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration Setup
st.set_page_config(page_title="Nifty Structural Core Stable", layout="wide")
st.title("🎯 Nifty 50 Mathematical Price-Density Variance Engine")
st.write("De-trending Nifty 50 (^NSEI) from May 1, 2026. Zero-division error patch deployed.")

# Fetch 1-Hour Accurate Nifty 50 Data safely
@st.cache_data(ttl=300)
def load_pure_data():
    df_raw = yf.download(tickers="^NSEI", start="2026-05-01", interval="1h")
    if df_raw.empty:
        return pd.DataFrame()
        
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    # Drop empty or corrupted data rows instantly
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    # 1. Column D: Date and Time Formatter
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    # 2. Column A: Exact Formula -> (High + Low) / 2
    df['Column A'] = (df['High'] + df['Low']) / 2.0
    
    # 3. Column B: Your Original Base Loop (Multiplier = 0.0001)
    multiplier_base = 0.0001
    n_col_b = np.zeros(len(df))
    
    if len(df) > 0:
        n_col_b[0] = float(df['Column A'].iloc[0])
        
    for i in range(1, len(df)):
        n_col_b[i] = n_col_b[i-1] + (multiplier_base * (float(df['Column A'].iloc[i]) - n_col_b[i-1]))
        
    df['Column B'] = n_col_b
    
    # 4. Column C: Exact Formula -> A - B Deviation
    df['Column C'] = df['Column A'] - df['Column B']
    
    # ---------------------------------------------------------
    # 🛠️ PURE MATHEMATICAL MICRO-DENSITY TRACKING ARRAYS
    # ---------------------------------------------------------
    # Micro-Range Dispersion Track
    df['Candle_Range'] = (df['High'] - df['Low']).astype(float)
    
    # Running Average Loop for Volatility Base (Multiplier = 0.01 for Micro-Breathing Space)
    multiplier_range = 0.01
    range_b = np.zeros(len(df))
    
    if len(df) > 0:
        range_b[0] = float(df['Candle_Range'].iloc[0]) if float(df['Candle_Range'].iloc[0]) > 0 else 1.0
        
    for i in range(1, len(df)):
        range_b[i] = range_b[i-1] + (multiplier_range * (float(df['Candle_Range'].iloc[i]) - range_b[i-1]))
        
    df['Range_B'] = range_b
    
    # Calculate the Core Density Compression Ratio (DCR) Safely
    # Using python lambda structure to completely eliminate "Script Execution Error" due to division
    df['DCR'] = df.apply(lambda row: float(row['Column C']) / float(row['Range_B']) if float(row['Range_B']) > 0.0001 else 0.0, axis=1)
    
    # ---------------------------------------------------------
    # 🎯 COLUMN E: PRICE-DENSITY VELOCITY TRAP INTERSECTION
    # ---------------------------------------------------------
    status_list = ["Baseline System Boot"]
    
    for i in range(1, len(df)):
        curr_c = float(df['Column C'].iloc[i])
        dcr = float(df['DCR'].iloc[i])
        
        # --- THE REFINED GEOMETRIC 90% VS 10% TRAP FILTER MATRIX ---
        if curr_c > 0:  # Column C is Plus (+)
            if dcr > 0.5:  # Organic structural expansion breakout -> 10% Real Move
                status_list.append("🟢 TRUE BULLISH MOMENTUM (DCR Expansion Verified)")
            else:  # Price artificially squeezed into a vacuum -> THE 90% CALL TRAP DETECTED!
                status_list.append("⚠️ 90% FAKE UPMOVE TRAP (Micro Density Compression!)")
                
        else:  # Column C is New/Minus (-)
            if dcr < -0.5:  # Organic structural collapse breakdown -> Real Melt
                status_list.append("🔴 TRUE BEARISH CRASH (DCR Expansion Verified)")
            else:  # Price artificially held within the density block -> THE 90% PUT TRAP DETECTED!
                status_list.append("⚠️ 90% FAKE DOWNMOVE TRAP (Micro Space Compression!)")
                
    df['Column E'] = status_list
    
    # Trim output display boundary cleanly for 1st May 2026 onwards
    df = df[df['Raw_Date'] >= '2026-05-01'].copy()
    
    # Reverse dataset array to throw latest 1-Hour rows at the top of the grid
    show_df = df[['Column D', 'Column A', 'Column B', 'Column C', 'Range_B', 'DCR', 'Column E']].copy()
    show_df = show_df.iloc[::-1]
    
    # Distinct Dashboard Layout Hex-Theming
    def color_trap_grid(val):
        if "🟢" in val: return 'background-color: #1fc07c; color: white; font-weight: bold;'
        if "🔴" in val: return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        if "⚠️" in val: return 'background-color: #d35400; color: white; font-weight: bold;'
        if "💤" in val: return 'background-color: #34495e; color: #bdc3c7;'
        return ''

    # Precision Decimal Formatter Block
    st.dataframe(show_df.style.format({
        'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
        'Range_B': '{:.4f}', 'DCR': '{:.4f}'
    }).map(color_trap_grid, subset=['Column E']), use_container_width=True)
else:
    st.error("Data ingestion failure: Data stream structure couldn't be mounted on Streamlit.")
