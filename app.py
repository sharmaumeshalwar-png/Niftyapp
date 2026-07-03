import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Kalman 0.50 VIX 25-Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Double Kalman [0.50 VIX 25-Candle Engine]")
st.write("🎯 **Aapki Custom Setting:** Fixed 25-Candle VIX Window + VIX 3-Regime Filtering + Kalman Price + Past 25-Candle Target + Double Kalman Smooth Momentum (P=0.50)")

# =====================================================================
# MATHEMATICAL ENGINE (Adaptive Noise Kalman Filter)
# =====================================================================
def apply_kalman_filter_adaptive(data_array, regimes_array, initial_p=50.0):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    
    filtered_values = []
    for z, regime in zip(data_array, regimes_array):
        # Strictly adaptive noise selection based on VIX regime
        if regime == 2:    # High Volatility (Panic Mode)
            q = 0.005      # High process noise (highly responsive)
            r = 0.05       # Low measurement noise (trusts immediate price action)
        elif regime == 0:  # Low Volatility (Quiet Mode)
            q = 0.0005     # Low process noise (smooth)
            r = 0.2        # Filters minor structural noise
        else:              # Normal Volatility (Standard Mode)
            q = 0.001
            r = 0.1
            
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Aligning Double Kalman & VIX 25-Candle Microstructure Matrices..."):
    # Download raw assets data (2 Years Window)
    raw_df = yf.download("BTC-USD", period="2y", interval="1h")
    vix_df = yf.download("^VIX", period="2y", interval="1d") # Daily VIX matching timeline
    
    if len(raw_df) == 0 or len(vix_df) == 0:
        st.error("YFinance API Timeout. Please refresh the dashboard.")
        st.stop()
        
    # MultiIndex Framework Elimination for BTC
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            if isinstance(raw_df[col], pd.DataFrame):
                df[col] = raw_df[col].iloc[:, 0]
            else:
                df[col] = raw_df[col]

    df.index = pd.to_datetime(df.index)
    
    # MultiIndex Framework Elimination for VIX & Mapping onto Hourly DataFrame
    vix_clean = vix_df['Close'].iloc[:, 0] if isinstance(vix_df['Close'], pd.DataFrame) else vix_df['Close']
    vix_clean.index = pd.to_datetime(vix_clean.index).date
    
    df['Date_Only'] = df.index.date
    df['VIX'] = df['Date_Only'].map(vix_clean).fillna(method='ffill').fillna(20.0) # Fallback baseline
    df.drop(columns=['Date_Only'], inplace=True)

    # 🔥 AAPKI CORE REQUIREMENT: Strictly Fixed 25-Candle Rolling Feature for VIX
    df['VIX_Rolling_25'] = df['VIX'].rolling(window=25).mean().fillna(20.0)

    # STRICT VIX 3-REGIME DEFINITION 
    # Regime 0: Quiet (<15), Regime 1: Normal (15-25), Regime 2: Panic (>25)
    df['VIX_Regime'] = np.where(df['VIX'] < 15, 0, np.where(df['VIX'] <= 25, 1, 2))

    # Base Matrix Definition (Price Kalman 1 Active)
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_adaptive(df['a_Close'].values, df['VIX_Regime'].values, initial_p=50.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']  # Pure (Close - Kalman)
    
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Past 25-Candle Target for BTC Direction
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    # Matrix inputs updated with VIX parameters & Fixed VIX 25 Window
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity', 'VIX_Regime', 'VIX_Rolling_25']
    df.dropna(subset=features_matrix + ['Target'], inplace=True)

# =====================================================================
# DYNAMIC SPLIT ENGINE (Strict 50:50 Ratio)
# =====================================================================
split_idx = int(len(df) * 0.50)

df_train = df.iloc[:split_idx]
X_train = df_train[features_matrix].copy()
y_train = df_train['Target'].copy()

df_predict = df.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix].copy()

if len(X_predict) == 0:
    st.error("Prediction matrix error. Waiting for market data ticks...")
else:
    # RandomForest Model Training
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3,            
        min_samples_leaf=1,     
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    # Raw Probabilities Prediction
    probabilities = model_flow.predict_proba(X_predict)
    
    df_predict['Prob_Down'] = probabilities[:, 0]
    df_predict['Prob_Up'] = probabilities[:, 1]

    # =====================================================================
    # LIVE TREND-LOCK CIRCUIT (DOUBLE KALMAN SIGNAL WITH VIX REGIME OVERRIDES)
    # =====================================================================
    final_signals = []
    scores_log = []
    raw_weighted_momentum_log = [] 
    current_state = "HOLD"
    
    accumulator = 0
    MAX_BUCKET = 5     
    MIN_BUCKET = -5    

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df
