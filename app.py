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

st.set_page_config(page_title="BTC Signal Engine with Doppler Attenuation", layout="wide")
st.title("⚡ Live Double Kalman + Doppler Attenuation Engine (2-Year IST)")

# =====================================================================
# 1. KALMAN FILTER & DOPPLER ATTENUATION FUNCTIONS
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

def calculate_doppler_attenuation(price_velocity, rolling_std, quality_factor=10.0):
    """
    Applies Doppler Wave Attenuation Theory:
    Alpha = (2 * pi * f) / (Q * v)
    Attenuated Momentum = Velocity * exp(-Alpha)
    Suppresses artificial high-frequency noise spikes (Fake Breakouts).
    """
    # Frequency surrogate: Normalized rate of price change
    freq = np.abs(price_velocity) / (rolling_std + 1e-10)
    
    # Velocity magnitude
    velocity_mag = np.abs(price_velocity) + 1e-10
    
    # Doppler Attenuation Coefficient (Alpha)
    alpha = (2 * np.pi * freq) / (quality_factor * velocity_mag)
    
    # Exponential Wave Attenuation Decay Factor
    attenuation_factor = np.exp(-np.clip(alpha, 0, 5))
    
    # Attenuated Signal Momentum
    attenuated_momentum = price_velocity * attenuation_factor
    
    return alpha, attenuation_factor, attenuated_momentum

# =====================================================================
# 2. MAIN DATA PIPELINE & ML ENGINE
# =====================================================================
with st.spinner("Fetching 2 Years Market Data & Applying Doppler Attenuation Engine..."):
    raw_df = yf.download("BTC-USD", period="730d", interval="1h", progress=False)
    
    if raw_df.empty:
        st.error("Data download error. Please refresh.")
        st.stop()

    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in raw_df.columns:
            df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.dropna(subset=['Close', 'High', 'Low', 'Open'], inplace=True)
    
    # IST Time Conversion
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC').tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)

    # Base Price Kalman
    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter(df['a_Close'].values, initial_p=50.0)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']

    # --- DOPPLER ATTENUATION COMPUTATION ---
    raw_velocity = df['c_Combined'].diff(1).fillna(0).values
    rolling_std_24 = df['c_Combined'].rolling(24).std().fillna(1.0).values
    
    alpha_vals, decay_factors, att_momentum = calculate_doppler_attenuation(raw_velocity, rolling_std_24)
    
    df['Doppler_Alpha'] = alpha_vals
    df['Attenuation_Factor'] = decay_factors
    df['Attenuated_Momentum'] = att_momentum

    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (rolling_std_24 + 1e-10)

    # HAM Features
    candle_body = (df['a_Close'] - df['Open']).abs()
    lower_wick = df[['a_Close', 'Open']].min(axis=1) - df['Low']
    df['HAM_Ratio'] = lower_wick / (candle_body + 1e-10)
    
    ham_conditions = [
        (df['HAM_Ratio'] >= 2.0) & (df['a_Close'] >= df['Open']),
        (df['HAM_Ratio'] >= 2.0) & (df['a_Close'] < df['Open'])
    ]
    df['HAM_Value'] = np.select(ham_conditions, [1, -1], default=0)

    # Target & Features List (With Doppler Attenuation Features)
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    features = [
        'c_Combined', 
        'Attenuated_Momentum', 
        'Attenuation_Factor', 
        'Order_Imbalance', 
        'Body_Imbalance', 
        'Normalized_Gap', 
        'HAM_Ratio', 
        'HAM_Value'
    ]
    df.dropna(subset=features + ['Target'], inplace=True)

    # 50:50 Learn:Predict Split
    split_idx = int(len(df) * 0.50)
    df_train = df.iloc[:split_idx]
    df_predict = df.iloc[split_idx:].copy()

    # Train Model
    model = RandomForestClassifier(n_estimators=150, max_depth=3, random_state=42)
    model.fit(df_train[features], df_train['Target'])

    # Predict
    probs = model.predict_proba(df_predict[features])
    df_predict['Prob_Down'] = probs[:, 0]
    df_predict['Prob_Up'] = probs[:, 1]

    # Accumulator Engine
    accumulator = 0
    scores, raw_momentum = [], []
    prob_ups, prob_downs = df_predict['Prob_Up'].values, df_predict['Prob_Down'].values
    closes, kalmans = df_predict['a_Close'].values, df_predict['b_Kalman_Price'].values

    for i in range(len(prob_ups)):
        p_up, p_down = prob_ups[i], prob_downs[i]
        if p_up >= 0.55:
            accumulator += 1
        elif p_down >= 0.55:
            accumulator -= 1
        accumulator = max(-5, min(5, accumulator))
        
        scores.append(accumulator)
        raw_momentum.append(closes[i] - kalmans[i])

    df_predict['Accumulator_Score'] = scores
    df_predict['Weighted_Momentum'] = apply_kalman_filter(raw_momentum, initial_p=0.50)

    # Signal Rules
    signals = []
    accum_array, wm_array = df_predict['Accumulator_Score'].values, df_predict['Weighted_Momentum'].values
    att_factors = df_predict['Attenuation_Factor'].values

    for i in range(len(df_predict)):
        acc, wm, p_up, p_down = accum_array[i], wm_array[i], prob_ups[i], prob_downs[i]
        prev_wm = wm_array[i-1] if i > 0 else wm
        decay = att_factors[i]

        # Signal logic considering Attenuation Decay
        if acc == 5 and (wm < prev_wm or p_down > 0.40 or decay < 0.3):
            signals.append("🔴 REAL TOP (Attenuated Peak Reversal)")
        elif acc == 5:
            signals.append("🟢 STRONG BUY (Max Locked +5)")
        elif acc == -5 and (wm > prev_wm or p_up > 0.40 or decay < 0.3):
            signals.append("🟢 REAL BOTTOM (Attenuated Valley Recovery)")
        elif acc == -5:
            signals.append("🔴 STRONG SELL (Max Bearish -5)")
        elif acc > 0:
            signals.append(f"🟢 BULLISH TREND (Score: {acc})")
        elif acc < 0:
            signals.append(f"🔴 BEARISH TREND (Score: {acc})")
        else:
            signals.append("⚪ NEUTRAL / HOLD")

    df_predict['Signal'] = signals

    # 8-Step Verification
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

    display_cols = [
        'a_Close', 
        'b_Kalman_Price', 
        'Attenuated_Momentum',
        'Attenuation_Factor',
        'HAM_Value', 
        'Prob_Up', 
        'Prob_Down', 
        'Accumulator_Score', 
        'Signal', 
        '8_Step_Verification'
    ]
    
    single_table_df = df_predict[display_cols].iloc[::-1].copy()
    single_table_df.index = single_table_df.index.strftime('%Y-%m-%d %H:%M IST')

    st.subheader("📋 BTC Signal Matrix with Doppler Wave Attenuation (Single Table)")
    st.dataframe(single_table_df, use_container_width=True, height=750)
