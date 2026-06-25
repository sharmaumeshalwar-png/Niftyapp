import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# STEP 1: UI THEME CONFIGURATION
st.set_page_config(page_title="Coppock-Engine 2026 Perfect", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #050b14 !important;
            color: #e2e8f0 !important;
        }
        h1, h3, p, span, label { color: #f8fafc !important; }
        .discovery-block {
            background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #10b981;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="discovery-block">
        <h1>🌌 The Coppock & Column C Discovery Engine</h1>
        <p><b>Computational Core:</b> Dual-Vector Convergence | <b>Window Lock:</b> Jan 2025 - Dec 2026</p>
    </div>
""", unsafe_allow_html=True)

# SAFE CHECK FOR PLOTLY
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False

# STEP 2: PRECISION CONTROLS SIDEBAR
st.sidebar.subheader("🔬 Precision Controls")
ticker = st.sidebar.text_input("Target Ticker", value="BTC-USD")
delta_lookback = st.sidebar.number_input("Delta Lookback (Column C)", min_value=1, value=14)
long_roc = st.sidebar.number_input("Coppock Long ROC", min_value=1, value=14)
short_roc = st.sidebar.number_input("Coppock Short ROC", min_value=1, value=11)
wma_smoothing = st.sidebar.number_input("Coppock WMA Smoothing", min_value=1, value=10)

run_sync = st.sidebar.button("🔄 Execute 2026 Handshake Loop")

# STEP 3: ARRAYS & DATA MATRIX
if 'coppock_db' not in st.session_state:
    st.session_state.coppock_db = pd.DataFrame()

# DUMMY DATAFRAME GENERATION FOR DEFAULT PREVIEW (ANTI-CRASH SAFETY)
if len(st.session_state.coppock_db) == 0 or run_sync:
    try:
        import yfinance as yf
        raw_feed = yf.download(tickers=str(ticker), interval="1h", period="2y", progress=False)
        is_synthetic = False
    except ModuleNotFoundError:
        is_synthetic = True
        st.sidebar.warning("⚠️ 'yfinance' module missing on server! Running in Safe-Synthetic Mode.")
        raw_feed = pd.DataFrame()

    if raw_feed.empty:
        is_synthetic = True
        date_range = pd.date_range(start="2024-06-01", end="2026-12-31", freq="1h")
        np.random.seed(42)
        simulated_close = 60000 + np.cumsum(np.random.normal(0, 150, len(date_range)))
        raw_feed = pd.DataFrame({
            'Open': simulated_close - 50, 'High': simulated_close + 100,
            'Low': simulated_close - 100, 'Close': simulated_close
        }, index=date_range)
        raw_feed.index.name = 'Datetime'

    # STEP 4 & 5: VECTOR CONVERGENCE ENGINE
    raw_close = raw_feed['Close'].values.flatten()
    raw_open = raw_feed['Open'].values.flatten()
    raw_high = raw_feed['High'].values.flatten()
    raw_low = raw_feed['Low'].values.flatten()
    raw_dates = raw_feed.index.to_numpy()
    
    total = len(raw_close)
    col_c = np.zeros(total)
    coppock_curve = np.zeros(total)
    time_list = []

    for t in range(total):
        time_list.append(pd.to_datetime(raw_dates[t]).strftime('%d %b %Y %H:%M'))
        if t >= int(delta_lookback):
            col_c[t] = raw_close[t] - raw_close[t - int(delta_lookback)]
            
    # Coppock Math
    roc_l = np.zeros(total)
    roc_s = np.zeros(total)
    for t in range(total):
        if t >= int(long_roc) and raw_close[t - int(long_roc)] != 0:
            roc_l[t] = ((raw_close[t] - raw_close[t - int(long_roc)]) / raw_close[t - int(long_roc)]) * 100
        if t >= int(short_roc) and raw_close[t - int(short_roc)] != 0:
            roc_s[t] = ((raw_close[t] - raw_close[t - int(short_roc)]) / raw_close[t - int(short_roc)]) * 100
            
    matrix_sum = roc_l + roc_s
    w_len = int(wma_smoothing)
    weights = np.arange(1, w_len + 1)
    w_sum = weights.sum()
    
    for t in range(total):
        if t >= w_len - 1:
            coppock_curve[t] = np.dot(matrix_sum[t - w_len + 1 : t + 1], weights) / w_sum

    # STEP 6 & 7: EXCEL CLASSIFICATION & MATRIX ISOLATION
    excel_zones = ["⚠️ Extreme Overbought (+2.5)" if x >= 2.5 else "🟢 Extreme Oversold (-2.5)" if x <= -2.5 else "Normal Zone" for x in coppock_curve]

    df_final = pd.DataFrame({
        'Date_Time': time_list, 'Column A (Raw Close)': raw_close, 'Column C (Delta Variance)': col_c,
        'Coppock Curve': coppock_curve, 'Excel Market Zone': excel_zones,
        'Open_Raw': raw_open, 'High_Raw': raw_high, 'Low_Raw': raw_low
    })

    df_final['Datetime_Parsed'] = pd.to_datetime(df_final['Date_Time'], format='%d %b %Y %H:%M')
    df_isolated = df_final[(df_final['Datetime_Parsed'] >= pd.Timestamp("2025-01-01")) & (df_final['Datetime_Parsed'] <= pd.Timestamp("2026-12-31"))].copy()
    
    df_isolated['Coppock_Prev'] = df_isolated['Coppock Curve'].shift(1)
    df_isolated['Bullish_Hint'] = (df_isolated['Column C (Delta Variance)'] > 0) & (df_isolated['Coppock Curve'] > 0) & (df_isolated['Coppock_Prev'] <= 0)
    df_isolated['Bearish_Hint'] = (df_isolated['Column C (Delta Variance)'] < 0) & (df_isolated['Coppock Curve'] < 0) & (df_isolated['Coppock_Prev'] >= 0)

    st.session_state.coppock_db = df_isolated.drop(columns=['Datetime_Parsed']).reset_index(drop=True)

# STEP 8: RENDER GRID
output = st.session_state.coppock_db.copy()

if not output.empty:
    st.write(f"### 📊 Step 8 Matrix: **{len(output)} Data Blocks Hard-Locked**")
    
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=output['Date_Time'], open=output['Open_Raw'], high=output['High_Raw'], low=output['Low_Raw'], close=output['Column A (Raw Close)'], name="Price Engine"))
        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Note: Install 'plotly' via requirements.txt to view the interactive charts.")

    st.write("#### 📋 All Possible Outcome Dates Matrix Logs")
    st.dataframe(output.iloc[::-1][['Date_Time', 'Column A (Raw Close)', 'Column C (Delta Variance)', 'Coppock Curve', 'Excel Market Zone']], use_container_width=True)
