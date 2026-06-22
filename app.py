import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. PREMIUM DARK BASE CONFIGURATION
st.set_page_config(page_title="Nifty Column-L Sign Matrix", layout="wide")

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
        <h1>🎯 Nifty 50 Pure Cascade (Column L Target Black Grid)</h1>
        <p><b>Column K:</b> (Sign of F) - (Sign of H) | <b>Column L:</b> (I row 2) - (I row 1)<br>
        <b>Visual Rule:</b> Column L cell turns <b>Absolute Black</b> whenever its mathematical sign changes from the previous row.</p>
    </div>
""", unsafe_allow_html=True)

# 2. DATA PIPELINE WITH TRAILING PAD BUFFER
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
    
    # 3. CORE 5-STAGE MATHEMATICAL CASCADE (Step 1 to 8 Counting verification)
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
    
    # 4. COLUMN K MATH (Sign F - Sign H)
    sign_f = np.sign(df['Column F'].values).astype(float)
    sign_h = np.sign(df['Column H'].values).astype(float)
    df['Column K'] = sign_f - sign_h
    
    # 5. COLUMN L COMPU-MATH: Strict Forward Differential (Row 2 - Row 1)
    df['Column L'] = df['Column I'].diff().astype(float)
    df['Column L'] = df['Column L'].fillna(0.0)
    
    # 🛠️ 6. COLUMN L DYNAMIC SIGN REVERSAL TRACKER (Vector Mapping)
    # Checking real-time when current L sign switches relative to previous row's L sign
    l_sign = np.sign(df['Column L'].values)
    l_blackout = np.zeros(total_rows, dtype=bool)
    
    for i in range(1, total_rows):
        # If previous row L has a valid sign and it doesn't match current row sign -> Reversal happened!
        if l_sign[i] != l_sign[i-1] and l_sign[i-1] != 0 and l_sign[i] != 0:
            l_blackout[i] = True
            
    df['L_Is_Black'] = l_blackout
    
    # 7. SIGNAL STATUS GENERATOR
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

    # 8. FILTER & STYLING INVERSION INTERFACE (Jan 1, 2025 Layout)
    df_filtered = df[df['Raw_Date'] >= '2025-01-01'].copy()
    
    if not df_filtered.empty:
        # We include 'L_Is_Black' as metadata for mapping colors, but format it cleanly
        show_df = df_filtered[['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'L_Is_Black', 'Signal_Status']].copy()
        show_df = show_df.iloc[::-1]  # Latest candle on top
        
        # Grid Custom Styles: Base styling for signals
        def style_signal_cells(val):
            if "🟢" in val: return 'background-color: #064e3b; color: #34d399; font-weight: bold;'
            if "🔴" in val: return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            if "⚠️" in val: return 'background-color: #b45309; color: #fef08a; font-weight: bold;'
            return 'background-color: #1f2937; color: #d1d5db;'

        # 🛠️ ADVANCED TARGETED COLUMN L COLOR MATRIX
        # Applies absolute black background strictly to L cells where sign change is flagged true
        def style_column_l(row):
            styles = [''] * len(row)
            l_index = row.index.get_loc('Column L')
            is_black_flag = row['L_Is_Black']
            
            if is_black_flag:
                styles[l_index] = 'background-color: #000000 !important; color: #ffffff !important; font-weight: bold; border: 1px solid #3b82f6;'
            else:
                styles[l_index] = 'background-color: #1f2937; color: #cbd5e1;'
            return styles

        # Render Table securely into Streamlit container
        st.dataframe(
            show_df.style.format({
                'Column A': '{:.2f}', 'Column B': '{:.4f}', 'Column C': '{:.4f}', 
                'Column E': '{:.4f}', 'Column F': '{:.4f}', 'Column G': '{:.4f}',
                'Column H': '{:.4f}', 'Column I': '{:.4f}', 'Column J': '{:.4f}', 
                'Column K': '{:.0f}', 'Column L': '{:.6f}'
            })
            .map(style_signal_cells, subset=['Signal_Status'])
            .apply(style_column_l, axis=1),
            column_order=['Column D', 'Column A', 'Column B', 'Column C', 'Column E', 'Column F', 'Column G', 'Column H', 'Column I', 'Column J', 'Column K', 'Column L', 'Signal_Status'],
            use_container_width=True
        )
    else:
        st.warning("January 1, 2025 filtered range empty.")
else:
    st.error("Data pipeline load error.")
