import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BTC Native Volatility 0.50 Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Double Kalman [Pure Crypto Volatility Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Native BTC Volatility (No External VIX) + 25-Candle Window + VIX 3-Regime + Double Kalman Smooth Momentum (P=0.50)")

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
        # Strictly adaptive noise selection based on Crypto VIX regime
        if regime == 2:    # High Volatility (Crypto Panic / Breakout)
            q = 0.005      
            r = 0.05       
        elif regime == 0:  # Low Volatility (Quiet accumulation)
            q = 0.0005     
            r = 0.2        
        else:              # Normal Volatility (Standard Crypto Mode)
            q = 0.001
            r = 0.1
            
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Processing Native Bitcoin Volatility & Double Kalman Matrices..."):
    # Download raw BTC asset data (2 Years Window)
    raw_df = yf.download("BTC-USD", period="2y", interval="1h")
    
    if len(raw_df) == 0:
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
    df['a_Close'] = df['Close']

    # 🔥 GENERATING NATIVE BITCOIN VOLATILITY (Pure Crypto VIX)
    # Annualized rolling hourly volatility mapped directly as a native feature
    log_returns = np.log(df['a_Close'] / df['a_Close'].shift(1))
    df['VIX'] = log_returns.rolling(window=24).std() * np.sqrt(24 * 365) * 100
    df['VIX'] = df['VIX'].fillna(30.0) # Baseline crypto fallback

    # Strictly Fixed 25-Candle Rolling Window for Crypto Volatility
    df['VIX_Rolling_25'] = df['VIX'].rolling(window=25).mean().fillna(30.0)

    # STRICT CRYPTO VOLATILITY 3-REGIME DEFINITION 
    # Regime 0: Quiet (<20), Regime 1: Normal (20-45), Regime 2: Crypto Panic/Breakout (>45)
    df['VIX_Regime'] = np.where(df['VIX'] < 20, 0, np.where(df['VIX'] <= 45, 1, 2))

    # Base Matrix Definition (Price Kalman 1 Active)
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
    
    # Matrix inputs updated with Native Crypto Volatility Parameters
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
    # LIVE TREND-LOCK CIRCUIT (DOUBLE KALMAN SIGNAL WITH BTC REGIME OVERRIDES)
    # =====================================================================
    final_signals = []
    scores_log = []
    raw_weighted_momentum_log = [] 
    current_state = "HOLD"
    
    accumulator = 0
    MAX_BUCKET = 5     
    MIN_BUCKET = -5    

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()
    vix_regimes = df_predict['VIX_Regime'].to_numpy()

    for i in range(len(prob_ups)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]
        c_val = closes[i]
        k_price_val = kalmans_price[i]
        current_regime = vix_regimes[i]

        # STRICT REGIME SENSITIVITY OVERRIDES FOR CRYPTO METRICS
        if current_regime == 2:      # Crypto Panic Regime
            barrier = 0.62
        elif current_regime == 0:    # Crypto Quiet Zone
            barrier = 0.52
        else:                        # Balanced Crypto Mode
            barrier = 0.55

        # Raw Accumulator Calculation
        if p_up >= barrier:
            accumulator += 1  
        elif p_down >= barrier:
            accumulator -= 1  
        
        accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))
        scores_log.append(accumulator)

        # Raw Weighted Momentum (Close - Kalman)
        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append(calc_raw_weighted)

        regime_label = "💥 PANIC/SPIKE" if current_regime == 2 else ("⚖️ NORMAL" if current_regime == 1 else "😴 QUIET")

        if accumulator == MAX_BUCKET:
            current_state = "BUY"
            final_signals.append(f"🟢 STRONG BUY ({regime_label} [5/5])")
            
        elif accumulator == MIN_BUCKET:
            current_state = "SELL"
            final_signals.append(f"🔴 STRONG SELL ({regime_label} [-5/-5])")
            
        else:
            if current_state == "BUY":
                if accumulator > 0:
                    final_signals.append(f"🟢 HOLD BUY | Points Decreasing ({regime_label} Score: {accumulator})")
                else:
                    final_signals.append(f"⚠️ BUY CRITICAL | Reversal Warning ({regime_label} Score: {accumulator})")
                    
            elif current_state == "SELL":
                if accumulator < 0:
                    final_signals.append(f"🔴 HOLD SELL | Points Increasing ({regime_label} Score: {accumulator})")
                else:
                    final_signals.append(f"⚠️ SELL CRITICAL | Reversal Warning ({regime_label} Score: {accumulator})")
                    
            else:
                final_signals.append(f"⚪ NEUTRAL | Building Conviction ({regime_label} Score: {accumulator})")

    # Mapping secure array data back to pandas
    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

    # Dynamic Hyper-Responsive Smooth Layer applied strictly with initial_p=0.50
    df_predict['Weighted_Momentum'] = apply_kalman_filter_adaptive(df_predict['Raw_Weighted_Momentum'].values, df_predict['VIX_Regime'].values, initial_p=0.50)

    # Display Configuration
    clean_display_cols = ['VIX', 'VIX_Rolling_25', 'VIX_Regime', 'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
    display_df['VIX'] = display_df['VIX'].round(2)
    display_df['VIX_Rolling_25'] = display_df['VIX_Rolling_25'].round(2)
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    
    # Sorting to get latest ticks on top
    display_df = display_df.sort_index(ascending=False)
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Bitcoin Engine (Pure BTC Native Volatility Bound Matrix)")
    st.dataframe(display_df, use_container_width=True, height=750)
