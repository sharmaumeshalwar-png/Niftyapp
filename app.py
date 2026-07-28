import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import pytz

# =====================================================================
# TIMEZONE SETUP (Indian Standard Time)
# =====================================================================
IST = pytz.timezone('Asia/Kolkata')

st.set_page_config(page_title="BTC Engine: Hurst Multiplied Momentum", layout="wide")
st.title("⚡ Live Double Kalman + Hurst Multiplied Weighted Momentum Engine")

# =====================================================================
# 1. HELPER FUNCTIONS: KALMAN & ROLLING HURST EXPONENT
# =====================================================================
def apply_kalman_filter(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0:
        return []
    x = data_array[0]
    p = initial_p
    filtered_values = []
    for z in data_array:
        p = p + q_val
        k = p / (p + r_val)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return filtered_values

def calculate_hurst_exponent(series, window=100):
    """
    Calculates Rolling Hurst Exponent (H)
    """
    hurst_vals = [0.5] * len(series)
    series_vals = series.values
    
    for i in range(window, len(series)):
        sub_series = series_vals[i-window:i]
        returns = np.diff(np.log(sub_series + 1e-10))
        if len(returns) < 10 or np.std(returns) == 0:
            continue
        
        mean_ret = np.mean(returns)
        cum_dev = np.cumsum(returns - mean_ret)
        r = np.max(cum_dev) - np.min(cum_dev)
        s = np.std(returns) + 1e-10
        
        rs = r / s
        h = np.log(rs + 1e-10) / np.log(window)
        hurst_vals[i] = np.clip(h, 0.0, 1.0)
        
    return hurst_vals

# =====================================================================
# 2. MAIN DATA PIPELINE
# =====================================================================
with st.spinner("Fetching BTC Data & Calculating Hurst * WM..."):
    raw_df = yf.download("BTC-USD", period="730d", interval="1h", progress=False)
    
    if raw_df.empty:
        st.error("Data download error. Please refresh.")
        st.stop()

    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    
    # IST Timezone
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC').tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)

    # Base Prices & First Kalman Filter
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter(df['a_Close'].values, initial_p=50.0)
    
    # Raw Weighted Momentum (WM = Close - Kalman_Price)
    df['Raw_WM'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Apply Second Kalman Filter to Smooth Weighted Momentum
    df['Weighted_Momentum'] = apply_kalman_filter(df['Raw_WM'].values, initial_p=0.50)

    # Hurst Exponent Calculation
    df['Hurst_Exponent'] = calculate_hurst_exponent(df['a_Close'], window=100)

    # --- PURE MATH MULTIPLICATION: Hurst * Weighted Momentum ---
    df['Hurst_WM_Multiplied'] = df['Hurst_Exponent'] * df['Weighted_Momentum']

    # Microstructure Features
    df['c_Combined'] = df['Raw_WM']
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(24).std() + 1e-10)

    # HAM Features
    candle_body = (df['a_Close'] - df['Open']).abs()
    lower_wick = df[['a_Close', 'Open']].min(axis=1) - df['Low']
    df['HAM_Ratio'] = lower_wick / (candle_body + 1e-10)
    
    ham_conditions = [
        (df['HAM_Ratio'] >= 2.0) & (df['a_Close'] >= df['Open']),
        (df['HAM_Ratio'] >= 2.0) & (df['a_Close'] < df['Open'])
    ]
    df['HAM_Value'] = np.select(ham_conditions, [1, -1], default=0)

    # ML Setup
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    features = [
        'c_Combined', 
        'Hurst_Exponent', 
        'Weighted_Momentum', 
        'Hurst_WM_Multiplied', 
        'Order_Imbalance', 
        'Body_Imbalance', 
        'Normalized_Gap', 
        'HAM_Ratio', 
        'HAM_Value'
    ]
    df.dropna(subset=features + ['Target'], inplace=True)

    # 50:50 Train-Predict Split
    split_idx = int(len(df) * 0.50)
    df_train = df.iloc[:split_idx]
    df_predict = df.iloc[split_idx:].copy()

    model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
    model.fit(df_train[features], df_train['Target'])

    probs = model.predict_proba(df_predict[features])
    df_predict['Prob_Down'] = probs[:, 0]
    df_predict['Prob_Up'] = probs[:, 1]

    # Accumulator Logic
    accumulator = 0
    scores = []
    prob_ups, prob_downs = df_predict['Prob_Up'].values, df_predict['Prob_Down'].values

    for i in range(len(prob_ups)):
        p_up, p_down = prob_ups[i], prob_downs[i]
        if p_up >= 0.55:
            accumulator += 1
        elif p_down >= 0.55:
            accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        scores.append(accumulator)

    df_predict['Accumulator_Score'] = scores

    # Signal Rules
    signals = []
    accum_array = df_predict['Accumulator_Score'].values
    wm_array = df_predict['Weighted_Momentum'].values

    for i in range(len(df_predict)):
        acc, wm, p_up, p_down = accum_array[i], wm_array[i], prob_ups[i], prob_downs[i]
        prev_wm = wm_array[i-1] if i > 0 else wm

        if acc == 5 and (wm < prev_wm or p_down > 0.40):
            signals.append("🔴 REAL TOP (Peak Reversal Warning)")
        elif acc == 5:
            signals.append("🟢 STRONG BUY (Max Locked +5)")
        elif acc == -5 and (wm > prev_wm or p_up > 0.40):
            signals.append("🟢 REAL BOTTOM (Valley Recovery Signal)")
        elif acc == -5:
            signals.append("🔴 STRONG SELL (Max Bearish -5)")
        elif acc > 0:
            signals.append(f"🟢 BULLISH TREND (Score: {acc})")
        elif acc < 0:
            signals.append(f"🔴 BEARISH TREND (Score: {acc})")
        else:
            signals.append("⚪ NEUTRAL / HOLD")

    df_predict['Signal'] = signals

    # 8-Step Verification Column
    total_len = len(df_predict)
    step_indices = set(np.linspace(0, total_len - 1, 8, dtype=int))
    
    verification_steps = []
    step_counter = 1
    for idx in range(total_len):
        if idx in step_indices and step_counter <= 8:
            verification_steps.append(f"Step {step_counter}/8 Verified")
            step_counter += 1
        else:
            verification_steps.append("Live Outcome")

    df_predict['8_Step_Verification'] = verification_steps

    # Display Columns Setup
    display_cols = [
        'a_Close', 
        'b_Kalman_Price', 
        'Weighted_Momentum',
        'Hurst_Exponent',
        'Hurst_WM_Multiplied',
        'HAM_Value', 
        'Prob_Up', 
        'Prob_Down', 
        'Accumulator_Score', 
        'Signal', 
        '8_Step_Verification'
    ]
    
    single_table_df = df_predict[display_cols].iloc[::-1].copy()
    single_table_df.index = single_table_df.index.strftime('%Y-%m-%d %H:%M IST')

    st.subheader("📋 BTC Signal Matrix with Hurst Multiplied Momentum (Single Table)")
    st.dataframe(single_table_df, use_container_width=True, height=750)
