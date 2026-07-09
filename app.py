import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Nifty Dual Momentum Engine", layout="wide")
st.title("📊 Nifty 50 Live 1-Hour Hybrid Double Kalman [0.50 Engine]")
st.write("🎯 **Aapki Custom Setting:** Strictly 2-Year Data Window + Strictly 50:50 Engine Split + Net Probability Flow & Weighted Momentum Tracker")

# =====================================================================
# MATHEMATICAL ENGINE 1: LINEAR FILTER
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=100.0): 
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p  
    q = 0.0001     
    r = 2.5        
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(float(x))
    return filtered_values

# =====================================================================
# MATHEMATICAL ENGINE 2: NON-LINEAR FILTER
# =====================================================================
def apply_non_linear_kalman_momentum(data_array):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = 1.0  
    q = 0.05   
    r = 0.2    
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(float(x))
    return filtered_values

with st.spinner("Aligning 25-Candle Dual Kalman Nifty Microstructure Matrices..."):
    # STRICT 2-YEAR WINDOW WITH REPAIR LAYER
    raw_df = yf.download("^NSEI", period="2y", interval="1h", progress=False, repair=True)
    
    if raw_df.empty:
        st.error("🚨 YFinance Data Down. Please re-run or check internet connection.")
        st.stop()
        
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
        
    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].ffill()

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')

    df['Prev_High'] = df['High'].shift(1).ffill().bfill()
    df['Prev_Low'] = df['Low'].shift(1).ffill().bfill()

    df['a_Close'] = df['Close'].astype(float)
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].to_numpy(), initial_p=100.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']  
    
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_matrix + ['Target']).copy()

# =====================================================================
# STRICT 50:50 ENGINE SPLIT LAYER
# =====================================================================
split_idx = int(len(df_clean) * 0.50)
df_train = df_clean.iloc[:split_idx].copy()
X_train = df_train[features_matrix]
y_train = df_train['Target']

df_predict = df_clean.iloc[split_idx:].copy()
X_predict = df_predict[features_matrix]

if len(X_predict) == 0 or len(X_train) == 0:
    st.error(f"⚠️ Insufficient Data rows for strict 50:50 split execution.")
else:
    model_flow = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    raw_prob_down = probabilities[:, 0]
    raw_prob_up = probabilities[:, 1]

    # =====================================================================
    # BAYESIAN MEMORY FLOW ENGINE (STRICT SUM = 1.0)
    # =====================================================================
    corrected_prob_up_list = []
    corrected_prob_down_list = []
    corrected_sum_list = []
    
    final_signals = []
    scores_log = []
    raw_weighted_momentum_log = [] 
    trap_status_log = [] 
    cum_prob_flow_log = []
    flow_direction_log = []
    
    current_state = "HOLD"
    accumulator = 0
    
    current_cum_up = float(raw_prob_up[0]) if len(raw_prob_up) > 0 else 0.5
    current_cum_down = float(raw_prob_down[0]) if len(raw_prob_down) > 0 else 0.5
    decay = 0.70 

    closes = df_predict['a_Close'].to_numpy()
    kalmans_price = df_predict['b_Kalman_Price'].to_numpy()
    prev_highs = df_predict['Prev_High'].to_numpy()
    prev_lows = df_predict['Prev_Low'].to_numpy()

    for i in range(len(raw_prob_up)):
        p_up_raw = float(raw_prob_up[i]) if not np.isnan(raw_prob_up[i]) else 0.5
        p_down_raw = float(raw_prob_down[i]) if not np.isnan(raw_prob_down[i]) else 0.5
        c_val = float(closes[i])
        k_price_val = float(kalmans_price[i])
        p_high = float(prev_highs[i])
        p_low = float(prev_lows[i])

        if i == 0:
            current_cum_up = p_up_raw
            current_cum_down = p_down_raw
        else:
            current_cum_up = (current_cum_up * decay) + (p_up_raw * (1 - decay))
            current_cum_down = (current_cum_down * decay) + (p_down_raw * (1 - decay))
        
        total_sum = current_cum_up + current_cum_down + 1e-10
        current_cum_up /= total_sum
        current_cum_down /= total_sum
        
        corrected_prob_up_list.append(current_cum_up)
        corrected_prob_down_list.append(current_cum_down)
        corrected_sum_list.append(current_cum_up + current_cum_down)

        net_prob_flow = current_cum_up - current_cum_down
        cum_prob_flow_log.append(net_prob_flow)
        
        if i == 0:
            flow_direction_log.append("🔄 START")
        else:
            if net_prob_flow > cum_prob_flow_log[i-1]:
                flow_direction_log.append("📈 FLOW RISING")
            elif net_prob_flow < cum_prob_flow_log[i-1]:
                flow_direction_log.append("📉 FLOW FALLING")
            else:
                flow_direction_log.append("⚖️ FLAT FLOW")

        if current_cum_up >= 0.52: 
            accumulator += 1  
        elif current_cum_down >= 0.52: 
            accumulator -= 1  
        accumulator = max(-5, min(5, accumulator))
        scores_log.append(int(accumulator))

        calc_raw_weighted = c_val - k_price_val
        raw_weighted_momentum_log.append(calc_raw_weighted)

        trap_msg = "TREND VALID"

        if accumulator == 5:
            current_state = "BUY"
            if c_val > p_high: 
                final_signals.append("🟢 STRONG BUY (Max [5/5])")
            else:
                final_signals.append("❌ NO ENTRY (Wait Breakout)")
                trap_msg = "⚠️ BULL TRAP"
        elif accumulator == -5:
            current_state = "SELL"
            if c_val < p_low: 
                final_signals.append("🔴 STRONG SELL (Max [-5/-5])")
            else:
                final_signals.append("🟢 HOLD LONG (No Short)")
                trap_msg = "⚠️ BEAR TRAP"
        else:
            if current_state == "BUY":
                if accumulator > 0: 
                    final_signals.append(f"🟢 HOLD BUY (Score: {accumulator})")
                else:
                    if c_val < p_low: 
                        final_signals.append(f"⚠️ BUY CRITICAL (Score: {accumulator})")
                    else:
                        final_signals.append(f"🔄 HOLD BUY | Fake Dip (Score: {accumulator})")
                        trap_msg = "⚠️ BEAR TRAP INSIDE"
            elif current_state == "SELL":
                if accumulator < 0: 
                    final_signals.append(f"🔴 HOLD SELL (Score: {accumulator})")
                else:
                    if c_val > p_high: 
                        final_signals.append(f"⚠️ SELL CRITICAL (Score: {accumulator})")
                    else:
                        final_signals.append(f"🔄 HOLD SELL | Fake Pump (Score: {accumulator})")
                        trap_msg = "⚠️ BULL TRAP INSIDE"
            else:
                final_signals.append(f"⚪ NEUTRAL (Score: {accumulator})")

        trap_status_log.append(trap_msg)

    # State Assignment Layer
    df_predict['Prob_Up'] = np.array(corrected_prob_up_list, dtype=float)
    df_predict['Prob_Down'] = np.array(corrected_prob_down_list, dtype=float)
    df_predict['Prob_Sum'] = np.array(corrected_sum_list, dtype=float)
    
    df_predict['d_ML_Signal'] = np.array(final_signals, dtype=object)
    df_predict['Trap_Status'] = np.array(trap_status_log, dtype=object)
    df_predict['Accumulator_Score'] = np.array(scores_log, dtype=int)
    df_predict['Raw_Weighted_Momentum'] = np.array(raw_weighted_momentum_log, dtype=float)
    df_predict['Net_Prob_Flow'] = np.array(cum_prob_flow_log, dtype=float)
    df_predict['Flow_State'] = np.array(flow_direction_log, dtype=object)

    # Core Kalman-filtered Weighted Momentum Engine
    df_predict['Weighted_Momentum'] = apply_kalman_filter_custom(df_predict['Raw_Weighted_Momentum'].to_numpy(), initial_p=0.50)
    non_linear_filtered = apply_non_linear_kalman_momentum(df_predict['Weighted_Momentum'].to_numpy())
    df_predict['Step_Momentum'] = np.round(non_linear_filtered)

    # Presentation Output Creation
    display_df = pd.DataFrame(index=df_predict.index)
    display_df['a_Close'] = df_predict['a_Close'].round(2)
    display_df['Prob_Up'] = df_predict['Prob_Up'].round(3)
    display_df['Prob_Down'] = df_predict['Prob_Down'].round(3)
    display_df['Prob_Sum'] = df_predict['Prob_Sum'].round(2)
    display_df['Net_Prob_Flow'] = df_predict['Net_Prob_Flow'].round(2)
    display_df['Weighted_Momentum'] = df_predict['Weighted_Momentum'].round(2) # Tracked strictly as requested
    display_df['Step_Momentum'] = df_predict['Step_Momentum'].astype(int)
    display_df['Flow_State'] = df_predict['Flow_State']
    display_df['Accumulator_Score'] = df_predict['Accumulator_Score'].astype(int)
    display_df['d_ML_Signal'] = df_predict['d_ML_Signal']
    display_df['Trap_Status'] = df_predict['Trap_Status']
    
    display_df = display_df.sort_index(ascending=False)
    
    string_dates = display_df.index.to_series().dt.strftime('%Y-%m-%d %H:%M').values
    display_df.index = string_dates
    display_df.index.name = "Date (IST)"

    st.subheader(f"📋 Live 1-Hour Nifty Dual Momentum Engine Dashboard")
    st.dataframe(display_df, use_container_width=True, height=750)
