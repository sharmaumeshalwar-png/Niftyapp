import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. PREMIUM DARK BASE CONFIGURATION
st.set_page_config(page_title="Nifty Fixed Column L Engine", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0b0f19 !important;
            color: #ffffff !important;
        }
        h1, h3, p, span, label { color: #ffffff !important; }
        .title-block {
            background: linear-gradient(90deg, #111827, #1f2937);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-block">
        <h1>🎯 Nifty 50 Pure Cascade (Fixed Column I & L Parsing)</h1>
        <p><b>Column K:</b> (Sign of F) - (Sign of H) | <b>Column L:</b> Strict Visual Row Math [Row 2 - Row 1]<br>
        <b>Visual Matrix Rule:</b> Sign Change = <b>Absolute Black Cell</b> + White Text. Otherwise = <b>Pure White Cell</b> + Black Text.</p>
    </div>
""", unsafe_allow_html=True)

# 2. DATA PIPELINE WITH TRAILING BUFFER
@st.cache_data(ttl=300)
def load_pure_data():
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
    
    # 3. CORE 5-STAGE MATHEMATICAL CASCADE (Step 1 to 8 Counting)
    df['Column A'] = ((df['High'] + df['Low']) / 2.0).astype(float)
    
    col_b = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_b[0] = float(df['Column A'].values[0])
    for i in range(1, total_rows):
        col_b[i] = col_b[i-1] + (multiplier * (float(df['Column A'].values[i]) - col_b[i-1]))
    df['Column B'] = col_b
    
    df['Column C'] = (df['Column A'] - df['Column B']).astype(float)
    
    col_e = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_e[0] = float(df['Column C'].values[0])
    for i in range(1, total_rows):
        col_e[i] = col_e[i-1] + (multiplier * (float(df['Column C'].values[i]) - col_e[i-1]))
    df['Column E'] = col_e
    
    df['Column F'] = (df['Column C'] - df['Column E']).astype(float)
    
    col_g = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_g[0] = float(df['Column F'].values[0])
    for i in range(1, total_rows):
        col_g[i] = col_g[i-1] + (multiplier * (float(df['Column F'].values[i]) - col_g[i-1]))
    df['Column G'] = col_g
    
    df['Column H'] = (df['Column F'] - df['Column G']).astype(float)
    
    # 🌟 COLUMN I ABSOLUTE CALCULATION LOCKED
    col_i = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_i[0] = float(df['Column H'].values[0])
    for i in range(1, total_rows):
        col_i[i] = col_i[i-1] + (multiplier * (float(df['Column H'].values[i]) - col_i[i-1]))
    df['Column I'] = col_i
    
    col_j = np.zeros(total_rows, dtype=float)
    if total_rows > 0: col_j[0] = float(col_i[0])
    for i in range(1, total_rows):
        col_j[i] = col_j[i-1] + (multiplier * (col_i[i] - col_j[i-1]))
    df['Column J'] = col_j
    
    # 4. COLUMN K CALCULATOR
    sign_f = np.sign(df['Column F'].values).astype(float)
    sign_h = np.sign(df['Column H'].values).astype(float)
    df['Column K'] = sign_f - sign_h
    
    # SIGNAL STATUS GENERATOR
    status_list = ["System Booting"]
    for i in range(1, total_rows):
        curr_k = float(df['Column K'].values[i])
        curr_c = float(df['Column C'].values[i])
        if curr_c > 0:
            if curr_k == 2.0 or curr_k == 0.0:  status_list.append("🟢 SIGN BULLISH")
            else:  status_list.append("⚠️ 90% CALL TRAP")
        else:
            if curr_k == -2.0 or curr_k == 0.0:  status_list.append("🔴 SIGN BEARISH")
            else:  status_list.append("⚠️ 90% PUT TRAP")
    df['Signal_Status'] = status_list

    # 5. FILTER & DISPLAY INVERSION (Latest data on top)
    df_filtered = df[df['Raw_Date'] >= '2025-01-01'].copy()
    
    if not df_filtered.empty:
        show_df = df_filtered[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Signal_Status']].copy()
        show_df = show_df.iloc[::-1].reset_index(drop=True)
        
        # 🛠️ 6. STRICT VISUAL COLUMN L CORRECTION ENGINE
        # Screen rows setup: Index '0' is Row 1 (Latest), Index '1' is Row 2 (Previous)
        display_rows = len(show_df)
        col_l_visual = np.zeros(display_rows, dtype=float)
        
        for i in range(display_rows - 1):
            # Formula: (Row 2 of Column I) - (Row 1 of Column I)
            # In inverted array: Index i+1 is visual Row 2, Index i is visual Row 1
            col_l_visual[i] = float(show_df.loc[i+1, 'Column I']) - float(show_df.loc[i, 'Column I'])
            
        show_df['Column L'] = col_l_visual

        # 🛠️ 7. PURE VISUAL SIGN REVERSAL ENGINE
        l_styles = [''] * display_rows
        
        # Scan from bottom to top of the display grid to check consecutive sign shifts
        for i in range(display_rows - 2, -1, -1):
            val_current = col_l_visual[i]      # Current cell
            val_previous = col_l_visual[i+1]   # Historically previous cell (one row below on screen)
            
            # Check strict mathematical sign change (+ to - OR - to +)
            if (val_current > 0 and val_previous < 0) or (val_current < 0 and val_previous > 0):
                l_styles[i] = 'background-color: #000000 !important; color: #ffffff !important; font-weight: bold; border: 1px solid #3b82f6;'
            else:
                l_styles[i] = 'background-color: #ffffff !important; color: #000000 !important; font-weight: bold;'
                
        # Default fallback for the absolute bottom edge row
        if display_rows > 0:
            l_styles[-1] = 'background-color: #ffffff !important; color: #000000 !important; font-weight: bold;'
            
        show_df['L_Style_String'] = l_styles

        # 8. THEME RENDERING PIPELINE
        def style_signal_cells(val):
            if "🟢" in val: return 'background-color: #064e3b; color: #34d399; font-weight: bold;'
            if "🔴" in val: return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            if "⚠️" in val: return 'background-color: #b45309; color: #fef08a; font-weight: bold;'
            return 'background-color: #1f2937; color: #d1d5db;'

        def apply_direct_l_style(row):
            styles = [''] * len(row)
            l_index = row.index.get_loc('Column L')
            styles[l_index] = row['L_Style_String']
            return styles

        # Render complete customized grid
        st.dataframe(
            show_df.style.format({
                'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
                'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.4f}',
                'Column H': '{:.4f}', 'Column I': '{:.4f}', 'Column J': '{:.4f}', 
                'Column K': '{:.0f}', 'Column L': '{:.6f}'
            })
            .map(style_signal_cells, subset=['Signal_Status'])
            .apply(apply_direct_l_style, axis=1),
            column_order=['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'Signal_Status'],
            use_container_width=True
        )
    else:
        st.warning("January 1, 2025 filtered range empty.")
else:
    st.error("Data pipeline load error.")
