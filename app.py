import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.title("Nifty 50: Comprehensive Historical Terminal")
st.write("Data Scope: Continuous 1-Hour Candles Since Jan 2025")

# ==========================================
# STEP 1: RESTORE COMPLETE HISTORICAL STREAM
# ==========================================
@st.cache_data
def load_nifty_data_maximum():
    try:
        ticker = "^NSEI"
        # API parameters adjusted to force maximize data pipeline length for 1h granularity
        df = yf.download(ticker, start="2025-01-01", interval="1h", auto_adjust=True, threads=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        return df
    except Exception as e:
        st.error(f"Fallback triggered due to: {str(e)}")
        dates = pd.date_range(start="2025-01-01", periods=1500, freq="h")
        return pd.DataFrame({"Close": np.sin(np.linspace(0, 50, 1500)) * 300 + 23500}, index=dates)

raw_data = load_nifty_data_maximum()

# Row count indicator for live tracking
st.metric(label="🚀 Total 1-Hour Candles Loaded in System", value=len(raw_data))

if raw_data.empty:
    st.error("Engine failed to synchronize data pool.")
else:
    # ==========================================
    # STEP 2 & 3: ENGINE MODULE A & B (TRIX)
    # ==========================================
    raw_data['A'] = raw_data['Close']
    
    trix_period = 15
    log_a = np.log(raw_data['A'])
    ema1 = log_a.ewm(span=trix_period, adjust=False).mean()
    ema2 = ema1.ewm(span=trix_period, adjust=False).mean()
    ema3 = ema2.ewm(span=trix_period, adjust=False).mean()
    
    raw_data['B'] = np.exp(ema3)
    
    # ==========================================
    # STEP 4: ENGINE MODULE C (Residue Delta)
    # ==========================================
    raw_data['C'] = raw_data['A'] - raw_data['B']
    
    # ==========================================
    # STEP 5: ENGINE MODULE D (Strict Locked Supertrend)
    # ==========================================
    atr_period = 20
    multiplier = 4.0
    
    c_diff = raw_data['C'].diff().abs()
    atr_c = c_diff.rolling(window=atr_period).mean()
    
    hl2_c = raw_data['C'] 
    basic_ub = hl2_c + (multiplier * atr_c)
    basic_lb = hl2_c - (multiplier * atr_c)
    
    final_ub = np.zeros(len(raw_data))
    final_lb = np.zeros(len(raw_data))
    supertrend = np.zeros(len(raw_data))
    direction = np.zeros(len(raw_data)) 
    
    start_idx = atr_period
    for i in range(len(raw_data)):
        if i < start_idx:
            supertrend[i] = raw_data['C'].iloc[i]
            direction[i] = 1
            final_ub[i] = basic_ub.iloc[i]
            final_lb[i] = basic_lb.iloc[i]
            continue
            
        if basic_ub.iloc[i] < final_ub[i-1] or raw_data['C'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb.iloc[i] > final_lb[i-1] or raw_data['C'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        if direction[i-1] == 1 and raw_data['C'].iloc[i] < final_lb[i]:
            direction[i] = -1
            supertrend[i] = final_ub[i]
        elif direction[i-1] == -1 and raw_data['C'].iloc[i] > final_ub[i]:
            direction[i] = 1
            supertrend[i] = final_lb[i]
        else:
            direction[i] = direction[i-1]
            supertrend[i] = final_lb[i] if direction[i] == 1 else final_ub[i]
            
    raw_data['D'] = supertrend
    raw_data['ST_Dir'] = direction

    # ==========================================
    # STEP 6: ENGINE MODULE E (Action Signals)
    # ==========================================
    raw_data['E'] = "HOLD"
    raw_data.loc[raw_data['ST_Dir'] == 1, 'E'] = "🟢 BUY (Trend Locked)"
    raw_data.loc[raw_data['ST_Dir'] == -1, 'E'] = "🔴 SELL (Trend Locked)"

    # FIX: Purani multi-column dropna ki wajah se rows delete ho rahi thi, ab strict cleanup hoga
    raw_data = raw_data.dropna(subset=['B', 'C', 'D'])

    # ==========================================
    # STEP 7: MATRIX GRID SORTING (Latest First)
    # ==========================================
    output_matrix = raw_data[['A', 'B', 'C', 'D', 'E']].copy()
    
    output_matrix['A'] = output_matrix['A'].map(lambda x: f"{x:.2f}")
    output_matrix['B'] = output_matrix['B'].map(lambda x: f"{x:.2f}")
    output_matrix['C'] = output_matrix['C'].map(lambda x: f"{x:.4f}") 
    output_matrix['D'] = output_matrix['D'].map(lambda x: f"{x:.4f}")
    
    output_matrix = output_matrix.sort_index(ascending=False)

    # UI Table Rendering
    st.subheader("📋 1-Hour Complete Continuous Matrix")
    st.dataframe(output_matrix, use_container_width=True, height=600)
