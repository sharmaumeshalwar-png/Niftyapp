import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Dual Momentum Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Hybrid Double Kalman [0.50 Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly Only Nifty 50 Index Data + Strictly Past 25-Candle Window + Double Filtered Weighted Momentum Layer + Instant Linear & Non-Linear ML Column Arrays")

# =====================================================================
# MATHEMATICAL ENGINE 1: LINEAR FILTER (Price & Original Momentum Layer)
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0): 
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    
    # LINEAR DISTANCE PARAMETERS: Line ko price se door aur smooth rakhne ke liye
    q = 0.0001     # Process noise bohot kam kiya (Stable movement)
    r = 2.5        # Measurement noise badha diya (Gap creation layer)
    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

# =====================================================================
# MATHEMATICAL ENGINE 2: NON-LINEAR FILTER (For the New Step Column Only)
# =====================================================================
def apply_non_linear_kalman_momentum(data_array):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = 1.0  
    q = 0.05   # High process noise = Real-time snap reaction without lag
    r = 0.2    # Extremely low measurement noise = Instant breakout tracing
    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

with st.spinner("Aligning 25-Candle Dual Kalman Nifty Microstructure Matrices..."):
    # Fail-safe data download to avoid Blank Data issue
    raw_df = yf.download("^NSEI", period="1y", interval="1h", group_by='column')
    
    if raw_df.empty:
        raw_df = yf.download("^NSEI", period="1mo", interval="1h")
        
    if raw_df.empty:
        st.error("YFinance API Timeout or Indian Market Closed. Please refresh the dashboard.")
        st.stop()
        
    # MultiIndex Framework Elimination
    df = pd.DataFrame(index=raw_df.index)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            if isinstance(raw_df[col], pd.DataFrame):
                df[col] = raw_df[col].iloc[:, 0].ffill()
            else:
                df[col] = raw_df[col].ffill()

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition (Price Kalman 1 Active - With New Safe Distance)
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=100.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']  # Pure (Close - Kalman) With Proper Gap
    
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']

    # STRICT PAST 25-CANDLE TARGET WINDOW
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_matrix + ['Target']).copy()

    # FIXING THE WINDOW TO STRICTLY PAST 25 CANDLES
    if len(df_clean) >= 25:
        df_clean = df_clean.tail(25)

# =====================================================================
# DYNAMIC SPLIT ENGINE (Strict 50:50 Ratio on 25 Candles)
# =====================================================================
split_idx = int(len(df_clean) * 0.50)

df_train = df_clean.iloc[:split_idx].copy()
X_train = df_train[features_matrix]
y_train = df_train['Target']

df_predict = df_clean.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix]

if len(X_predict) == 0 or len(X_train) == 0:
    st.error(f"⚠️ Data size insufficient for split. Total rows: {len(df_clean)}")
else:
    # 🎯 FIXING THE CLASS VARIANCE CRASH: Strict fallback verification check
    if len(np.unique(y_train)) > 1:
        model_flow = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model_flow.fit(X_train, y_train)
        probabilities = model_flow.predict_proba(X_predict)
        df_predict['Prob_Down'] = probabilities[:, 0]
        df_predict['Prob_Up'] = probabilities[:, 1]
    else:
        # Fallback setting if data trends only in single direction within 12 candles
        df_predict['Prob_Down'] = 0.50 if y_train.iloc[0] == 1 else 1.0
        df_predict['Prob_Up'] = 0.50 if y_train.iloc[0] == 0 else 1.0

    # Price Action Columns
    df_predict['Prev_High'] = df_predict['High'].shift(1)
    df_predict['Prev_Low'] = df_predict['Low'].shift(1)

    # =====================================================================
    # LIVE TREND-LOCK CIRCUIT WITH TRAP-DETECTION FILTER (HIGH/LOW BREAK)
    # =====================================================================
    final_signals = []
    scores_log = []
    raw_weighted_momentum_log = [] 
    trap_status_log = [] 
    current_state = "HOLD"
    accumulator = 0

    prob_ups = df_predict['Prob_Up'].to_numpy()
    prob_downs = df_predict['Prob_Down'].to_numpy()
    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()
    prev_highs = df_predict['Prev_High'].to_numpy()
    prev_lows = df_predict['Prev_Low'].to_numpy()

    for i in range(len(prob_ups)):
        p_up = prob_ups[i]
        p_down = prob_downs[i]
        c_val = closes[i]
        k_price_val = kalmans_price[i]
        p_high = prev_highs[i] if not np.isnan(prev_highs[i]) else c_val
        p_low = prev_lows[i] if not np.isnan(prev_lows[i]) else c_val

        # Accumulator Engine
        if p_up >= 0.55: accumulator += 1  
        elif p_down >= 0.55: accumulator -= 1  
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(accumulator)

        # Raw Weighted Momentum (Close - Smooth Kalman)
        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append(calc_raw_weighted)

        trap_msg = "TREND VALID"

        if accumulator == 5:
            current_state = "BUY"
            if c_val > p_high: final_signals.append("🟢 STRONG BUY (Max Locked [5/5])")
            else:
                final_signals.append("❌ NO ENTRY (Wait for Breakout)")
                trap_msg = "⚠️ BULL TRAP (High Not Broken)"
        elif accumulator == -5:
            current_state = "SELL"
            if c_val < p_low: final_signals.append("🔴 STRONG SELL (Max Locked [-5/-5])")
            else:
                final_signals.append("🟢 HOLD LONG (No Short Entry)")
                trap_msg = "⚠️ BEAR TRAP (Low Not Broken)"
        else:
            if current_state == "BUY":
                if accumulator > 0: final_signals.append(f"🟢 HOLD BUY | Points Decreasing (Score: {accumulator})")
                else:
                    if c_val < p_low: final_signals.append(f"⚠️ BUY CRITICAL | Reversal Warning (Score: {accumulator})")
                    else:
                        final_signals.append(f"🔄 HOLD BUY | Fake Dip (Score: {accumulator})")
                        trap_msg = "⚠️ BEAR TRAP INSIDE BULL TREND"
            elif current_state == "SELL":
                if accumulator < 0: final_signals.append(f"🔴 HOLD SELL | Points Increasing (Score: {accumulator})")
                else:
                    if c_val > p_high: final_signals.append(f"⚠️ SELL CRITICAL | Reversal Warning (Score: {accumulator})")
                    else:
                        final_signals.append(f"🔄 HOLD SELL | Fake Pump (Score: {accumulator})")
                        trap_msg = "⚠️ BULL TRAP INSIDE BEAR TREND"
            else:
                final_signals.append(f"⚪ NEUTRAL | Building Conviction (Score: {accumulator})")

        trap_status_log.append(trap_msg)

    # Mapping back to dataframe
    df_predict['d_ML_Signal'] = final_signals
    df_predict['Trap_Status'] = trap_status_log 
    df_predict['Accumulator_Score'] = scores_log 
    df_predict['Raw_Weighted_Momentum'] = raw_weighted_momentum_log 

    # 1. 🟢 ORIGINAL LINEAR WEIGHTED MOMENTUM (Keep Decimals)
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].values, initial_p=0.50)

    # 2. 🆕 NEW NON-LINEAR STANDALONE STEP MOMENTUM COLUMN
    non_linear_filtered = apply_non_linear_kalman_momentum(df_predict['Weighted_Momentum'].values)
    df_predict['Step_Momentum'] = np.round(non_linear_filtered)

    # =====================================================================
    # 🔥 HIGH-SPEED ARRAY ML ON THE 25-CANDLE MOMENTUM TARGET MATRIX
    # =====================================================================
    df_predict['WM_Velocity'] = df_predict['Weighted_Momentum'].diff(1).fillna(0)
    df_predict['WM_Acceleration'] = df_predict['WM_Velocity'].diff(1).fillna(0)
    
    wm_features = ['Weighted_Momentum', 'WM_Velocity', 'WM_Acceleration']
    df_predict['WM_Target'] = np.where(df_predict['Weighted_Momentum'] > df_predict['Weighted_Momentum'].shift(1), 1, 0)
    
    X_wm = df_predict[wm_features]
    y_wm = df_predict['WM_Target']
    
    # Second ML level multi-variance protection loop
    if len(np.unique(y_wm)) > 1:
        # Linear Wave Classifier 
        model_linear_batch = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=42, n_jobs=-1)
        model_linear_batch.fit(X_wm, y_wm)
        df_predict['ML_WM_Linear_Prob'] = model_linear_batch.predict_proba(df_predict[wm_features])[:, 1]
        
        # Non-Linear Structural Classifier
        model_nonlinear_batch = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        model_nonlinear_batch.fit(X_wm, y_wm)
        df_predict['ML_WM_NonLinear_Prob'] = model_nonlinear_batch.predict_proba(df_predict[wm_features])[:, 1]
    else:
        df_predict['ML_WM_Linear_Prob'] = 0.50 if (len(y_wm) > 0 and y_wm.iloc[0] == 1) else 0.0
        df_predict['ML_WM_NonLinear_Prob'] = 0.50 if (len(y_wm) > 0 and y_wm.iloc[0] == 1) else 0.0

    # Table View Layout Configuration
    clean_display_cols = [
        'a_Close', 'b_Kalman_Price', 'Prev_High', 'Prev_Low', 
        'Prob_Up', 'Prob_Down', 'Accumulator_Score', 
        'Weighted_Momentum', 'Step_Momentum', 
        'ML_WM_Linear_Prob', 'ML_WM_NonLinear_Prob',
        'd_ML_Signal', 'Trap_Status'
    ]
    
    display_df = df_predict[clean_display_cols].copy().sort_index(ascending=False)
        
    display_df['a_Close'] = display_df['a_Close'].round(2)
    display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
    display_df['Prev_High'] = display_df['Prev_High'].round(2)
    display_df['Prev_Low'] = display_df['Prev_Low'].round(2)
    display_df['Prob_Up'] = display_df['Prob_Up'].round(3)
    display_df['Prob_Down'] = display_df['Prob_Down'].round(3)
    display_df['Accumulator_Score'] = display_df['Accumulator_Score'].astype(int)
    display_df['Weighted_Momentum'] = display_df['Weighted_Momentum'].round(2) 
    display_df['Step_Momentum'] = display_df['Step_Momentum'].astype(int) 
    display_df['ML_WM_Linear_Prob'] = display_df['ML_WM_Linear_Prob'].round(3)
    display_df['ML_WM_NonLinear_Prob'] = display_df['ML_WM_NonLinear_Prob'].round(3)
    
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

    st.subheader(f"📋 Live 1-Hour Nifty Dual Momentum Engine Dashboard (Strict 25-Candle Setup)")
    st.dataframe(display_df, use_container_width=True)
