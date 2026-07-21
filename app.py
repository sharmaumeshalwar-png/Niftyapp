import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="BTC HA EMA Relativistic Engine", layout="wide", initial_sidebar_state="collapsed")

st.title("⚡ BTC Heikin-Ashi Dual EMA Engine (E=mc² + Kalman 0.80)")
st.caption("1H HA Close & 1H EMA locked -> 15M HA Close & 15M EMA -> EMA Difference -> 15M EMA E=mc² -> Kalman Filter (0.80)")

# =====================================================================
# 1. HEIKIN-ASHI & KALMAN ENGINES
# =====================================================================
def compute_heikin_ashi(df_in):
    df_ha = df_in.copy()
    
    op = df_ha['Open'].to_numpy().flatten()
    hi = df_ha['High'].to_numpy().flatten()
    lo = df_ha['Low'].to_numpy().flatten()
    cl = df_ha['Close'].to_numpy().flatten()
    
    ha_close = (op + hi + lo + cl) / 4.0
    ha_open = np.zeros(len(df_in))
    ha_open[0] = (op[0] + cl[0]) / 2.0
    
    for i in range(1, len(df_in)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
        
    ha_high = np.maximum(hi, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lo, np.minimum(ha_open, ha_close))
    
    df_ha['HA_Open'] = ha_open
    df_ha['HA_High'] = ha_high
    df_ha['HA_Low'] = ha_low
    df_ha['HA_Close'] = ha_close
    
    return df_ha

def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.80):
    if len(data_array) == 0: return []
    x, p = data_array[0], initial_p  
    filtered_values = []
    for z in data_array:
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

def apply_e_mc_square_on_ema(ema_series, volume_series, close_series, window=20):
    vol = np.array(volume_series, dtype=float)
    price = np.array(close_series, dtype=float)
    ema_vals = np.array(ema_series, dtype=float)
    
    # 15M EMA Momentum (Rate of Change)
    ema_mom = pd.Series(ema_vals).diff().fillna(0.0).to_numpy()
    
    # Mass (m) = Volume / Rolling Volume Average
    vol_sma = pd.Series(vol).rolling(window=window, min_periods=1).mean().to_numpy()
    mass_m = np.where(vol_sma > 0, vol / vol_sma, 1.0)
    mass_m = np.clip(mass_m, 0.5, 3.0) 
    
    # Volatility / Speed of light (c) = Price Return Volatility
    price_returns = pd.Series(price).pct_change().fillna(0.0).to_numpy()
    volatility_c = pd.Series(price_returns).rolling(window=window, min_periods=1).std().to_numpy()
    mean_vol = np.mean(volatility_c) if np.mean(volatility_c) > 0 else 1e-8
    norm_c = volatility_c / mean_vol
    norm_c = np.clip(norm_c, 0.5, 2.5)
    
    # E = m * c^2 applied to 15M EMA Momentum
    energy_e = ema_mom * mass_m * (norm_c ** 2)
    return energy_e

# =====================================================================
# DATA INGESTION
# =====================================================================
with st.spinner("Fetching Live BTC Data..."):
    try:
        df_1h_raw = yf.download(tickers="BTC-USD", period="60d", interval="1h")
        df_15m_raw = yf.download(tickers="BTC-USD", period="14d", interval="15m")
        
        for d in [df_1h_raw, df_15m_raw]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            if d.index.tz is None:
                d.index = d.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                d.index = d.index.tz_convert('Asia/Kolkata')
    except Exception as e:
        st.error(f"🚨 Data Fetching Error: {e}")
        st.stop()

# Step 1 & Step 2: Compute Heikin-Ashi Close
df_1h = compute_heikin_ashi(df_1h_raw)
df_15m = compute_heikin_ashi(df_15m_raw)

# Step 3 & Step 4: Compute EMA (20) on HA Close for both 1H and 15M
df_1h['1H_EMA'] = df_1h['HA_Close'].ewm(span=20, adjust=False).mean()
df_15m['15M_EMA'] = df_15m['HA_Close'].ewm(span=20, adjust=False).mean()

# Step 6: E=mc² on 15M EMA
m15_ema = df_15m['15M_EMA'].to_numpy()
m15_vol = df_15m['Volume'].to_numpy()
m15_close = df_15m['Close'].to_numpy()

df_15m['15M_EMA_E_mc2'] = apply_e_mc_square_on_ema(m15_ema, m15_vol, m15_close, window=20)

# Step 7: Kalman Filter (0.80) on E=mc² output
df_15m['15M_EMA_E_mc2_Kalman_080'] = apply_kalman_filter_custom(
    df_15m['15M_EMA_E_mc2'].to_numpy(), initial_p=1.0, q_val=0.001, r_val=0.80
)

# =====================================================================
# SAFE ALIGNMENT & DIFFERENCE COMPUTATION
# =====================================================================
df_15m_reset = df_15m.reset_index()
df_1h_reset = df_1h[['HA_Close', '1H_EMA']].reset_index()

df_1h_reset.rename(columns={
    'HA_Close': '1H_HA_Close_Frozen',
    '1H_EMA': '1H_EMA_Frozen'
}, inplace=True)

merged_df = pd.merge_asof(
    df_15m_reset.sort_values('Datetime'),
    df_1h_reset.sort_values('Datetime'),
    on='Datetime',
    direction='backward'
)

merged_df.set_index('Datetime', inplace=True)
merged_df.ffill(inplace=True)
merged_df.bfill(inplace=True)

df_grid = merged_df.copy()

# Step 5: Difference between 1H EMA and 15M EMA
df_grid['EMA_Difference'] = df_grid['1H_EMA_Frozen'] - df_grid['15M_EMA']

# Renaming for clarity
df_grid.rename(columns={'HA_Close': '15M_HA_Close'}, inplace=True)

latest = df_grid.iloc[-1]
latest_time = df_grid.index[-1].strftime('%Y-%m-%d %H:%M IST')

# =====================================================================
# 📊 DISPLAY MATRIX
# =====================================================================
st.markdown("---")
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("1H HA Close", f"${latest['1H_HA_Close_Frozen']:,.2f}")
m2.metric("15M HA Close", f"${latest['15M_HA_Close']:,.2f}")
m3.metric("1H EMA", f"${latest['1H_EMA_Frozen']:,.2f}")
m4.metric("15M EMA", f"${latest['15M_EMA']:,.2f}")
m5.metric("EMA Diff (1H-15M)", f"{latest['EMA_Difference']:.2f}")
m6.metric("15M EMA E=mc²", f"{latest['15M_EMA_E_mc2']:.4f}")
m7.metric("Kalman (0.80)", f"{latest['15M_EMA_E_mc2_Kalman_080']:.4f}")

st.markdown("---")

st.subheader("📋 Sequential Pipeline Table")

final_cols = [
    '1H_HA_Close_Frozen', '15M_HA_Close', 
    '1H_EMA_Frozen', '15M_EMA', 
    'EMA_Difference', '15M_EMA_E_mc2', '15M_EMA_E_mc2_Kalman_080'
]

display_df = df_grid[final_cols].copy()

display_df.rename(columns={
    '1H_HA_Close_Frozen': '1. 1H HA Close',
    '15M_HA_Close': '2. 15M HA Close',
    '1H_EMA_Frozen': '3. 1H EMA',
    '15M_EMA': '4. 15M EMA',
    'EMA_Difference': '5. EMA Diff (1H-15M)',
    '15M_EMA_E_mc2': '6. 15M EMA E=mc²',
    '15M_EMA_E_mc2_Kalman_080': '7. Kalman (0.80) Output'
}, inplace=True)

display_df = display_df.iloc[::-1]
display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M IST')

st.dataframe(display_df, use_container_width=True, height=650)
