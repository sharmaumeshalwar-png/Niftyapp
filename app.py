import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Pure Crypto VIX 0.50 Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live 1-Hour Double Kalman [Pure Crypto VIX Engine]")
st.write("🎯 **Aapki Custom Setting:** BTC Price + Native Crypto VIX Index Sync + Fixed 25-Candle Window + 3-Regime Dynamic Barriers + Double Kalman Smooth Momentum (P=0.50)")

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
        if regime == 2:    # Hyper Crypto Liquid Expansion (High Volatility)
            q = 0.005      # Ultra fast tracking for massive breakouts
            r = 0.05       
        elif regime == 0:  # Dry Compression Area (Low Volatility)
            q = 0.0005     # Smooth out chop noise
            r = 0.2        
        else:              # Standard Balance Zone
            q = 0.001
            r = 0.1
            
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Building Native Crypto Volatility & Price Matrices..."):
    # Download raw BTC data (2 Years Window)
    raw_df = yf.download("BTC-USD", period="2y", interval="1h")
    
    if len(raw_df) == 0:
        st.error("YFinance API Timeout. Please refresh the dashboard.")
        st.stop()
        
    # MultiIndex Framework Elimination
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            if isinstance(raw_df[col], pd.DataFrame):
                df[col] = raw_df[col].iloc[:, 0]
            else:
                df[col] = raw_df[col]

    df.index = pd.to_datetime(df.index)
    df['a_Close'] = df['Close']
    
    # =====================================================================
    # MATHEMATICAL ENGINE: PURE NATIVE CRYPTO VIX CALCULATION (25-Candle Structure)
    # =====================================================================
    # Step A: True Range Calculation
    df['H_L'] = df['High'] - df['Low']
    df['H_PC'] = np.abs(df['High'] - df['a_Close'].shift(1))
    df['L_PC'] = np.abs(df['Low'] - df['a_Close'].shift(1))
    df['True_Range'] = df[['H_L', 'H_PC', 'L_PC']].max(axis=1)
    
    # Step B: Rolling ATR Over Fixed 25-Candle Window
    df['Crypto_ATR_25'] = df['True_Range'].rolling(window=25).mean()
    
    # Step C: Standard Deviation Volatility of Returns Over Fixed 25-Candle Window
    df['Log_Ret'] = np.log(df['a_Close'] / df['a_Close'].shift(1))
    df['Crypto_Std_25'] = df['Log_Ret'].rolling(window=25).std() * np.sqrt(24 * 365) * 100 # Annualized Intraday Scale
    
    # Step D: Synthetic Pure Crypto VIX Vector Composition
    df['Crypto_VIX'] = (df['Crypto_ATR_25'] / df['a_Close'] * 1000) + (df['Crypto_Std_25'] * 0.1)
    df['Crypto_VIX'] = df['Crypto_VIX'].ffill().fillna(25.0)

    # Strictly Fixed 25-Candle Rolling Feature for the Volatility Matrix
    df['VIX_Rolling_25'] = df['Crypto_VIX'].rolling(window=25).mean().ffill().fillna(25.0)

    # STRICT NATIVE CRYPTO VOLATILITY 3-REGIME MAPPING BASED ON PERCENTILES
    vix_low = df['Crypto_VIX'].quantile(0.30)
    vix_high = df['Crypto_VIX'].quantile(0.85)
    
    df['VIX_Regime'] = np.where(df['Crypto_VIX'] < vix_low, 0, np.where(df['Crypto_VIX'] <= vix_high, 1, 2))

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
    
    # Past 25-Candle Target for Machine Learning Direction System
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    # Clean temporary columns
    df.drop(columns=['H_L', 'H_PC', 'L_PC', 'True_Range', 'Log_Ret'], inplace=True, errors='ignore')
    
    # Matrix inputs updated with Crypto VIX Parameters
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
    st.error("Prediction matrix split tracking mismatch error.")
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
    # LIVE TREND-LOCK CIRCUIT (DOUBLE KALMAN SIGNAL WITH BARRIER OVERRIDES)
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

        # NATIVE CRYPTO DYNAMIC OVERRIDES FOR ACTIVATION BARRIERS
        if current_regime == 2:      # Hyper Expansion Expansion Panic
            barrier = 0.62           # Rigid entry block to bypass false volatility spikes
        elif current_regime == 0:    # Dry Compression Area
            barrier = 0.52           # Sensitive trigger for breakout validation
        else:                        # Balanced Trade Zone
            barrier = 0.55

        # Raw Accumulator Score Calculations
        if p_up >= barrier:
            accumulator += 1  
        elif p_down >= barrier:
            accumulator -= 1  
        
        accumulator = max(MIN_BUCKET, min(MAX_BUCKET, accumulator))
        scores_log.append(accumulator)

        # Raw Weighted Momentum (Close - Kalman)
        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append(calc_raw_weighted)

        regime_label = "💥 CRYPTO PANIC" if current_regime == 2 else ("⚖️ NORMAL" if current_regime == 1 else "😴 COMPRESSION")

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

    # Mapping secure array data back to pandas dataframe
    df_predict['d_ML_Signal'] = final_signals
    df_predict['Accumulator_Score'] = scores_log  
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

    # Dynamic Hyper-Responsive Smooth Layer applied strictly with initial_p=0.50
    df_predict['Weighted_Momentum'] = apply_kalman_filter_adaptive(df_predict['Raw_Weighted_Momentum'].values, df_predict['VIX_Regime'].values, initial_p=0.50)

    # Display Configuration
    df_predict['Crypto_VIX'] = df_predict['Crypto_VIX'].round(2)
    clean_display_cols = ['Crypto_VIX', 'VIX_Rolling_25', 'VIX_Regime', 'a_Close', 'b_Kalman_Price', 'Prob_Up', 'Prob_Down', 'Accumulator_Score', 'Weighted_Momentum', 'd_ML_Signal']
    display_df = df_predict[clean_display_cols].copy()
    
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

    st.subheader(f"📋 Live 1-Hour BTC Matrix (Native Crypto Volatility VIX Architecture)")
    st.dataframe(display_df, use_container_width=True, height=750)
