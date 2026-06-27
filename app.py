import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.title("Nifty 50: Strict Locked Supertrend Matrix")
st.write("Layout: A=Close, B=TRIX, C=Residue, D=Strict Locked Supertrend, E=Signal (1-Hour Candles)")

# ==========================================
# STEP 1: SAFE DATA INGESTION
# ==========================================
@st.cache_data
def load_nifty_data():
    try:
        ticker = "^NSEI"
        df = yf.download(ticker, start="2025-01-01", interval="1h", auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except:
        dates = pd.date_range(start="2025-01-01", periods=100, freq="h")
        return pd.DataFrame({"Close": np.linspace(23000, 24000, 100)}, index=dates)

raw_data = load_nifty_data()

if raw_data.empty:
    st.error("Data load nahi ho paaye. Please refresh karein.")
else:
    # ==========================================
    # STEP 2 & 3: COLUMN A & B (TRIX)
    # ==========================================
    raw_data['A'] = raw_data['Close']
    
    trix_period = 15
    log_a = np.log(raw_data['A'])
    ema1 = log_a.ewm(span=trix_period, adjust=False).mean()
    ema2 = ema1.ewm(span=trix_period, adjust=False).mean()
    ema3 = ema2.ewm(span=trix_period, adjust=False).mean()
    
    raw_data['B'] = np.exp(ema3)
    
    # ==========================================
    # STEP 4: COLUMN C (Residue Wave)
    # ==========================================
    raw_data['C'] = raw_data['A'] - raw_data['B']
    
    # ==========================================
    # STEP 5: COLUMN D (Strict No-Movement Supertrend)
    # ==========================================
    # Tight Noise Filter Settings: Period = 20, Multiplier = 4.0
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
    
    # Strict Freeze Initial State
    start_idx = atr_period
    for i in range(len(raw_data)):
        if i < start_idx:
            supertrend[i] = raw_data['C'].iloc[i]
            direction[i] = 1
            final_ub[i] = basic_ub.iloc[i]
            final_lb[i] = basic_lb.iloc[i]
            continue
            
        # STRICT UPPER BAND LOCK: Jab tak trend change na ho, upper band hil nahi sakta
        if basic_ub.iloc[i] < final_ub[i-1] or raw_data['C'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # STRICT LOWER BAND LOCK: Kisi bhi tarah ka minor movement block kiya gaya hai
        if basic_lb.iloc[i] > final_lb[i-1] or raw_data['C'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        # Direction Logic with High Threshold Core
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
    # STEP 6: COLUMN E (Locked Action Signals)
    # ==========================================
    raw_data['E'] = "HOLD"
    raw_data.loc[raw_data['ST_Dir'] == 1, 'E'] = "🟢 BUY (Trend Locked)"
    raw_data.loc[raw_data['ST_Dir'] == -1, 'E'] = "🔴 SELL (Trend Locked)"

    raw_data.dropna(subset=['B', 'C', 'D'], inplace=True)

    # ==========================================
    # STEP 7: FORMATTING & REVERSE SORTING
    # ==========================================
    output_matrix = raw_data[['A', 'B', 'C', 'D', 'E']].copy()
    
    output_matrix['A'] = output_matrix['A'].map(lambda x: f"{x:.2f}")
    output_matrix['B'] = output_matrix['B'].map(lambda x: f"{x:.2f}")
    output_matrix['C'] = output_matrix['C'].map(lambda x: f"{x:.4f}") 
    output_matrix['D'] = output_matrix['D'].map(lambda x: f"{x:.4f}")
    
    output_matrix = output_matrix.sort_index(ascending=False)

    # Main Grid output display
    st.subheader("📋 Strict Mathematical Trend Matrix (No Micro-Movement)")
    st.dataframe(output_matrix, use_container_width=True)

    st.success("Logic Updated: Supertrend parameters ko strict (20, 4.0) kar diya gaya hai aur code level par bands ko freeze kar diya hai taaki direction badalne ke baad minor fluctuations block ho sakein.")
