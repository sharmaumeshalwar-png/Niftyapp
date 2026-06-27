import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.title("Nifty 50: TRIX-Supertrend Matrix Terminal")
st.write("Layout: A=Close, B=TRIX, C=Residue (Minus Allowed), D=Supertrend on C, E=Action Signal")

# ==========================================
# STEP 1: SAFE DATA INGESTION
# ==========================================
@st.cache_data
def load_nifty_data():
    try:
        ticker = "^NSEI"
        # 1-Hour candle data from Jan 2025
        df = yf.download(ticker, start="2025-01-01", interval="1h", auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except:
        dates = pd.date_range(start="2025-01-01", periods=100, freq="h")
        return pd.DataFrame({"Close": np.linspace(23000, 24000, 100), "High": 24050, "Low": 22950}, index=dates)

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
    # STEP 4: COLUMN C (Residue Wave - Minus Allowed)
    # ==========================================
    raw_data['C'] = raw_data['A'] - raw_data['B']
    
    # ==========================================
    # STEP 5: COLUMN D (Supertrend on C)
    # ==========================================
    # Supertrend Parameters: Period = 10, Multiplier = 3
    atr_period = 10
    multiplier = 3.0
    
    # Residue C ke upar True Range aur ATR calculate kar rahe hain
    # Chunki C ek single line matrix hai, iska change hi iska range hoga
    c_diff = raw_data['C'].diff().abs()
    atr_c = c_diff.rolling(window=atr_period).mean()
    
    # Basic Upper aur Lower Bands on Column C
    hl2_c = raw_data['C'] # Proxy for median price of residue
    basic_ub = hl2_c + (multiplier * atr_c)
    basic_lb = hl2_c - (multiplier * atr_c)
    
    # Final Bands aur Supertrend Matrix calculation
    final_ub = np.zeros(len(raw_data))
    final_lb = np.zeros(len(raw_data))
    supertrend = np.zeros(len(raw_data))
    direction = np.zeros(len(raw_data)) # 1 for Green/Up, -1 for Red/Down
    
    # Initialize first valid row
    start_idx = atr_period
    for i in range(len(raw_data)):
        if i < start_idx:
            supertrend[i] = raw_data['C'].iloc[i]
            direction[i] = 1
            continue
            
        # Upper Band logic
        if basic_ub.iloc[i] < final_ub[i-1] or raw_data['C'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Lower Band logic
        if basic_lb.iloc[i] > final_lb[i-1] or raw_data['C'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        # Direction aur Supertrend value assignment
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
    # STEP 6: COLUMN E (Supertrend Based Action Signal)
    # ==========================================
    raw_data['E'] = "HOLD"
    raw_data.loc[raw_data['ST_Dir'] == 1, 'E'] = "🟢 BUY (Residue Bullish)"
    raw_data.loc[raw_data['ST_Dir'] == -1, 'E'] = "🔴 SELL (Residue Bearish)"

    raw_data.dropna(subset=['B', 'C', 'D'], inplace=True)

    # ==========================================
    # STEP 7: FORMATTING & REVERSE SORTING
    # ==========================================
    output_matrix = raw_data[['A', 'B', 'C', 'D', 'E']].copy()
    
    output_matrix['A'] = output_matrix['A'].map(lambda x: f"{x:.2f}")
    output_matrix['B'] = output_matrix['B'].map(lambda x: f"{x:.2f}")
    output_matrix['C'] = output_matrix['C'].map(lambda x: f"{x:.4f}") 
    output_matrix['D'] = output_matrix['D'].map(lambda x: f"{x:.4f}")
    
    # Latest candle on top
    output_matrix = output_matrix.sort_index(ascending=False)

    # Display Matrix
    st.subheader("📋 TRIX-Supertrend Hybrid Matrix Matrix")
    st.dataframe(output_matrix, use_container_width=True)

    st.success("Logic Swap Complete: RAVI ko hata kar Column D mein true Supertrend (10, 3) lagaya gaya hai jo pure C variable par chal raha hai.")
