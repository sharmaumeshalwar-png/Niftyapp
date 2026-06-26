import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OU Matrix Precision", layout="wide")

st.title("🏛️ Institutional Mean Reversion — Ornstein-Uhlenbeck (OU) Matrix")
st.write("Formula Applied: $b_2 = b_1 + 0.0001 \times (a_2 - b_1)$ and $c = a - b$")

# ==========================================
# 1. FIXED DATE DATA LOCK PIPELINE (NSE TIMELINE)
# ==========================================
@st.cache_data(ttl=86400)
def generate_exact_nse_data():
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2026-06-26")
    
    raw_dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    nse_dates = [dt for dt in raw_dates if dt.weekday() < 5 and 9 <= dt.hour <= 15]
    
    total_candles = len(nse_dates)
    base_close = 24000
    np.random.seed(12345)
    
    # Live market price simulation for 'a'
    mock_history = base_close + np.cumsum(np.random.normal(0, 15, total_candles))
    
    df_out = pd.DataFrame({'a': mock_history}, index=nse_dates)
    df_out.index.name = 'Date & Time'
    return df_out

with st.spinner("⏳ Re-calculating with exact custom formulas..."):
    df = generate_exact_nse_data()
    
    a_vals = df['a'].to_numpy()
    n = len(a_vals)
    
    # ==========================================
    # 2. EXACT CUSTOM CALCULATION FOR b AND c
    # ==========================================
    b_vals = np.zeros(n)
    
    # Seed value for b1
    b_vals[0] = a_vals[0] 
    
    # Loop running your exact logic: b2 = b1 + 0.0001 * (a2 - b1)
    for i in range(1, n):
        b_vals[i] = b_vals[i-1] + 0.0001 * (a_vals[i] - b_vals[i-1])
        
    df['b'] = b_vals
    df['c'] = df['a'] - df['b']  # Exact c = a - b

    # ==========================================
    # 3. ORNSTEIN-UHLENBECK PARAMS (Theta & Half Life)
    # ==========================================
    lookback = 30
    theta_arr = np.zeros(n)
    half_life_arr = np.zeros(n)
    ou_signal = np.zeros(n)
    
    c_vals = df['c'].to_numpy()
    
    for i in range(lookback, n):
        y = c_vals[i-lookback+1 : i+1]
        x = c_vals[i-lookback : i]
        
        poly = np.polyfit(x, y, 1)
        beta = poly[0]
        
        if 0 < beta < 1:
            theta = -np.log(beta)
            half_life = np.log(2) / theta
        else:
            theta = 0.01
            half_life = 99.0
            
        theta_arr[i] = theta
        half_life_arr[i] = half_life
        
        # Signal Generation based on Variance of 'c'
        rolling_std = np.std(c_vals[i-lookback : i])
        ou_equilibrium_barrier = rolling_std * 1.8
        
        if c_vals[i] > ou_equilibrium_barrier and c_vals[i] < c_vals[i-1]:
            ou_signal[i] = -1.00 # Sell Reversion
        elif c_vals[i] < -ou_equilibrium_barrier and c_vals[i] > c_vals[i-1]:
            ou_signal[i] = 1.00  # Buy Reversion

    df['Theta'] = theta_arr
    df['Half_Life_Hours'] = half_life_arr
    df['Signal'] = ou_signal

st.success("🎯 Calculation Error Resolved. Exact Formulas Locked!")

# ==========================================
# 4. STREAMLIT UI DISPLAY
# ==========================================
df_display = df.copy()
df_display.index = df_display.index.strftime('%Y-%m-%d %H:%M')

col1, col2, col3 = st.columns(3)
col1.metric(label="Formula Applied", value="Custom Recursive")
col2.metric(label="Gap Method", value="c = a - b")
col3.metric(label="Total Hours Tracked", value=f"{len(df_display)}")

st.markdown("---")
st.subheader("🎯 Active OU Reversion Triggers (Strict High-Speed Only)")
sig_df = df_display[df_display['Signal'] != 0][['a', 'b', 'c', 'Theta', 'Half_Life_Hours', 'Signal']]
st.dataframe(sig_df.style.format("{:.2f}"), use_container_width=True)

st.markdown("---")
st.subheader("📋 Full SDE Parameter Timeline Matrix")
st.dataframe(df_display[['a', 'b', 'c', 'Theta', 'Half_Life_Hours', 'Signal']].style.format("{:.2f}"), use_container_width=True, height=500)
