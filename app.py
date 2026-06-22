import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. PREMIUM BLACK THEME CONFIGURATION (Injecting Custom CSS directly into the frame)
st.set_page_config(page_title="Nifty Advanced Black Cascade", layout="wide")

st.markdown("""
    <style>
        /* Main background and app alignment */
        .main { background-color: #0b0f19; color: #ffffff; }
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
        }
        h1, h3, p, span, label { color: #ffffff !important; }
        
        /* Custom header look */
        .title-block {
            background: linear-gradient(90deg, #1e293b, #0f172a);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Multi-Stage Cascade (F - H Matrix)</h1>
        <p><b>Column K:</b> (Sign of F) - (Sign of H) | <b>Column L:</b> (Current Row I) - (Previous Row I)<br>
        <b>Trend Change Rule:</b> Active monitoring when Price Movement opposes Column L Sign Vector.</p>
    </div>
""", unsafe_allow_html=True)

# 2. DATA PIPELINE WITH HISTORICAL PADDING
@st.cache_data(ttl=300)
def load_pure_data():
    # Fetching 2 years data to buffer smooth loop calculations from Jan 1, 2025
    df_raw = yf.download(tickers="^NSEI", period="2y", interval="1h")
    if df_raw.empty:
        return pd.DataFrame()
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    return df_raw

df = load_pure_data()

if not df.empty:
    df = df.reset_index()
    
    time_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    df['Raw_Date'] = pd.to_datetime(df[time_col])
    df['Column D'] = df['Raw_Date'].dt.strftime('%d %b %Y %H:%M')
    
    total_rows = len(df)
    multiplier = 0.0001
    
    # 3. 8-STEP VERIFICATION MATHEMATICAL BLOCKS
    # Step 1: Raw Price
    df['Column A'] = ((df['High'] + df['Low']) / 2.0).astype(float)
    
    # Step 2: Smooth Loop Stage 1
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (multiplier * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    # Step 3: First Deviation
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    # Step 4: Smooth Loop Stage 2
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (multiplier * (float(df['Column C'].values[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    # Step 5: Second Deviation
    df['Column F'] = (df['Column C'] - df['Column E']).astype(float)
    
    # Step 6: Smooth Loop Stage 3
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_g[0] = float(df['Column F'].values[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (multiplier * (float(df['Column F'].values[i]) - col_g[i-1]))
    df['Column G'] = col_g
    
    # Step 7: Third Deviation
    df['Column H'] = (df['Column F'] - df['Column G']).astype(float)
    
    # Step 8: Smooth Loop Stage 4
    col_i = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_i[0] = float(df['Column H'].values[0])
    for i in range(1, total_rows):
        col_i[i] = col_i[i-1] + (multiplier * (float(df['Column H'].values[i]) - col_i[i-1]))
    df['Column I'] = col_i
    
    # 4. COLUMN K & COLUMN L COMPUTATION ENGINE
    sign_f = np.sign(df['Column F'].values).astype(float)
    sign_h = np.sign(df['Column H'].values).astype(float)
    df['Column K'] = sign_f - sign_h
    
    # 🛠️ Column L: (Current row I) - (Previous row I)
    col_l = np.zeros(total_rows, dtype=float)
    for i in range(1, total_rows):
        col_l[i] = col_i[i] - col_i[i-1]
    df['Column L'] = col_l
    
    # 5. DYNAMIC TREND CHANGE OPPOSITION FILTER
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_l = float(df['Column L'].values[i])
        price_change = float(df['Column A'].values[i]) - float(df['Column A'].values[i-1])
        curr_k = float(df['Column K'].values[i])
        
        # Checking logic if Price action goes completely opposite to L's structural sign
        is_opposite = (curr_l > 0 and price_change < 0) or (curr_l < 0 and price_change > 0)
        
        if is_opposite:
            status_list.append("⚡ TREND CHANGE ALERT (Price Opposes L!)")
        else:
            if curr_k == 2.0 or curr_k == 0.0:
                status_list.append("🟢 BULLISH CONTINUATION")
            elif curr_k == -2.0 or curr_k == 0.0:
                status_list.append("🔴 BEARISH CONTINUATION")
            else:
                status_list.append("⚠️ GRID INVERSION")
                
    df['Signal_Status'] = status_list

    # 6. TIME BOUND PRESENTATION FILTER (Jan 1, 2025 Onwards)
    df_filtered = df[df['Raw_Date'] >= '2025-01-01'].copy()
    
    if not df_filtered.empty:
        show_df = df_filtered[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column H', 'Column I', 'Column K', 'Column L', 'Signal_Status']].copy()
        show_df = show_df.iloc[::-1]  # Latest candle on top
        
        # Premium Dark Theme Color Mapping Engine
        def style_dark_grid(val):
            if "⚡ TREND" in val: return 'background-color: #7b1113; color: #ffffff; font-weight: bold; border: 1px solid #f59e0b;'
            if "🟢" in val: return 'background-color: #064e3b; color: #34d399; font-weight: bold;'
            if "🔴" in val: return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            return 'background-color: #1e293b; color: #cbd5e1;'

        # Render Frame safely into Streamlit
        st.dataframe(show_df.style.format({
            'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
            'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column H': '{:.4f}', 
            'Column I': '{:.4f}', 'Column K': '{:.0f}', 'Column L': '{:.6f}'
        }).map(style_dark_grid, subset=['Signal_Status']), use_container_width=True)
    else:
        st.warning("January 1, 2025 filtered range empty.")
else:
    st.error("Data pipeline load error.")
