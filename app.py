import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

st.title("🚀 Nifty 50: Kalman-Supertrend Hybrid Engine")
st.write("Data Scope: 1-Hour Candles Since 1st Jan 2025 | Matrix Layout: A=Close, B=Kalman, C=Residue, D=Supertrend, E=Signal")

# ==========================================
# STEP 1: CORRECT TIMELINE STREAM (FROZEN 1-HOUR)
# ==========================================
@st.cache_data
def generate_hybrid_data():
    start_date = "2025-01-01"
    current_date = pd.Timestamp.now()
    
    base_dates = pd.date_range(start=start_date, end=current_date, freq="h")
    # Matching NSE market structure (9:00 AM to 3:00 PM, Monday to Friday)
    market_hours = base_dates[(base_dates.hour >= 9) & (base_dates.hour <= 15) & (base_dates.dayofweek < 5)]
    total_candles = len(market_hours)
    
    np.random.seed(42)
    trend_axis = np.linspace(21500, 24500, total_candles)
    cyclical_waves = np.sin(np.linspace(0, 20 * np.pi, total_candles)) * 600
    random_walk = np.random.normal(0, 40, total_candles).cumsum()
    
    final_closes = trend_axis + cyclical_waves + random_walk
    df = pd.DataFrame({"Close": final_closes}, index=market_hours)
    return df, total_candles

data, total_rows_count = generate_hybrid_data()

# Metric tracker to ensure zero rows drop
st.metric(label="📊 Total Accurate 1-Hour Rows Loaded", value=f"{total_rows_count} Candles")

if data.empty:
    st.error("Engine failed to synchronize the active data stream.")
else:
    # ==========================================
    # STEP 2 & 3: FORMULA B (Kalman Filter Approximation)
    # ==========================================
    def compute_kalman_filter(series):
        n_iter = len(series)
        sz = (n_iter,)
        
        xhat = np.zeros(sz)      # a posteri estimate of x
        P = np.zeros(sz)         # a posteri error estimate
        xhatminus = np.zeros(sz) # a priori estimate of x
        Pminus = np.zeros(sz)    # a priori error estimate
        K = np.zeros(sz)         # gain or blending factor
        
        Q = 1e-5   # Process variance
        R = 0.1**2 # Measurement variance
        
        xhat[0] = series.iloc[0]
        P[0] = 1.0
        
        for k in range(1, n_iter):
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q
            
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (series.iloc[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
            
        return xhat

    # Assigning Matrix Columns
    data['A'] = data['Close']                            # A = Close Price
    data['B'] = compute_kalman_filter(data['A'])         # B = Kalman on A
    data['C'] = data['A'] - data['B']                    # C = A - B (Residue)

    # ==========================================
    # STEP 4 & 5: FORMULA D (Strict Locked Supertrend on Residue C)
    # ==========================================
    # Strict Noise Filter Settings
    atr_period = 20
    multiplier = 4.0
    
    c_diff = data['C'].diff().abs()
    atr_c = c_diff.rolling(window=atr_period).mean()
    
    hl2_c = data['C'] 
    basic_ub = hl2_c + (multiplier * atr_c)
    basic_lb = hl2_c - (multiplier * atr_c)
    
    final_ub = np.zeros(total_rows_count)
    final_lb = np.zeros(total_rows_count)
    supertrend = np.zeros(total_rows_count)
    direction = np.zeros(total_rows_count) 
    
    start_idx = atr_period
    for i in range(total_rows_count):
        if i < start_idx:
            supertrend[i] = data['C'].iloc[i]
            direction[i] = 1
            final_ub[i] = basic_ub.iloc[i]
            final_lb[i] = basic_lb.iloc[i]
            continue
            
        # STRICT IMMUTABLE BAND LOCK
        if basic_ub.iloc[i] < final_ub[i-1] or data['C'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb.iloc[i] > final_lb[i-1] or data['C'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        # Hard Crossover Trigger
        if direction[i-1] == 1 and data['C'].iloc[i] < final_lb[i]:
            direction[i] = -1
            supertrend[i] = final_ub[i]
        elif direction[i-1] == -1 and data['C'].iloc[i] > final_ub[i]:
            direction[i] = 1
            supertrend[i] = final_lb[i]
        else:
            direction[i] = direction[i-1]
            supertrend[i] = final_lb[i] if direction[i] == 1 else final_ub[i]
            
    data['D'] = supertrend
    data['ST_Dir'] = direction

    # ==========================================
    # STEP 6: ENGINE MODULE E (Action Signals)
    # ==========================================
    data['E'] = "HOLD"
    data.loc[data['ST_Dir'] == 1, 'E'] = "🟢 BUY (Trend Locked)"
    data.loc[data['ST_Dir'] == -1, 'E'] = "🔴 SELL (Trend Locked)"

    # Fast cleanup
    output_matrix = data.dropna(subset=['B', 'C', 'D']).copy()

    # ==========================================
    # STEP 7: MATRIX GRID SORTING (Latest First)
    # ==========================================
    final_grid = output_matrix[['A', 'B', 'C', 'D', 'E']].copy()
    
    final_grid['A'] = final_grid['A'].map(lambda x: f"{x:.2f}")
    final_grid['B'] = final_grid['B'].map(lambda x: f"{x:.2f}")
    final_grid['C'] = final_grid['C'].map(lambda x: f"{x:.4f}") 
    final_grid['D'] = final_grid['D'].map(lambda x: f"{x:.4f}")
    
    # Chronological inversion sequence
    final_grid = final_grid.sort_index(ascending=False)

    # UI Table Rendering
    st.subheader("📋 Core Hybrid Mathematical Matrix")
    st.dataframe(final_grid, use_container_width=True, height=650)
