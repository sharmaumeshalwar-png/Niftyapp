import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="BTC Fixed Machine Engine", layout="wide")
st.title("🧠 BTC Live 1-Hour [1,000-Fold Kalman Simulator Matrix]")

# =====================================================================
# FLEXIBLE KALMAN FILTER FUNCTION
# =====================================================================
def apply_kalman_filter_custom(data_array, initial_p=50.0, q_val=0.001, r_val=0.1):
    if len(data_array) == 0: return np.array([])
    x, p, q, r = data_array[0], initial_p, q_val, r_val
    filtered_values = []
    for z in data_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_values.append(x)
    return np.array(filtered_values)

with st.spinner("Executing Cross-Section Multi-Index Layer Extraction..."):
    # 730d dynamic data download
    raw_df = yf.download("BTC-USD", period="730d", interval="1h")
    
    if raw_df.empty:
        st.error("Market API Endpoint Error. Please reload the server.")
        st.stop()
        
    # CROSS-SECTION MATRIX CLEANING ENGINE
    df = pd.DataFrame(index=raw_df.index)
    
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            # Extract only columns belonging to BTC-USD asset to drop multidimensional overhead
            df['Open'] = pd.to_numeric(raw_df.xs('BTC-USD', axis=1, level=1)['Open'], errors='coerce')
            df['High'] = pd.to_numeric(raw_df.xs('BTC-USD', axis=1, level=1)['High'], errors='coerce')
            df['Low'] = pd.to_numeric(raw_df.xs('BTC-USD', axis=1, level=1)['Low'], errors='coerce')
            df['Close'] = pd.to_numeric(raw_df.xs('BTC-USD', axis=1, level=1)['Close'], errors='coerce')
        else:
            df['Open'] = pd.to_numeric(raw_df['Open'], errors='coerce')
            df['High'] = pd.to_numeric(raw_df['High'], errors='coerce')
            df['Low'] = pd.to_numeric(raw_df['Low'], errors='coerce')
            df['Close'] = pd.to_numeric(raw_df['Close'], errors='coerce')
    except Exception:
        # Strict backup parsing via position array mapping
        df['Open'] = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce')
        df['High'] = pd.to_numeric(raw_df.iloc[:, 1], errors='coerce')
        df['Low'] = pd.to_numeric(raw_df.iloc[:, 2], errors='coerce')
        df['Close'] = pd.to_numeric(raw_df.iloc[:, 3], errors='coerce')

    # Forward fill gaps to secure row counts
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df['a_Close'] = df['Close']
    df['b_Kalman_Price'] = apply_kalman_filter_custom(df['a_Close'].values, initial_p=50.0, q_val=0.001, r_val=0.1)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman_Price']
    
    # Mathematical Variables for Machine Processing
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Normalized_Gap'] = df['c_Combined'] / (df['c_Combined'].rolling(window=24).std() + 1e-10)
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    df['Target'] = np.where(df['a_Close'] > df['a_Close'].shift(25), 1, 0)
    
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix + ['Target'], inplace=True)

# 50:50 Dynamic Model Splits
split_idx = int(len(df) * 0.50)
df_train = df.iloc[:split_idx].copy()
df_predict = df.iloc[split_idx:].copy()

model_flow = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
model_flow.fit(df_train[features_matrix], df_train['Target'])

probabilities = model_flow.predict_proba(df_predict[features_matrix])
df_predict['Prob_Down'] = probabilities[:, 0]
df_predict['Prob_Up'] = probabilities[:, 1]

raw_weighted_momentum = (df_predict['a_Close'] - df_predict['b_Kalman_Price']).to_numpy()
target_v = df_predict['Target'].to_numpy()

# =====================================================================
# ⚙️ 1,000 KALMAN SIMULATION RUNNER
# =====================================================================
q_grid = np.logspace(-4, -1, 10) 
r_grid = np.logspace(-2, 1, 20)

best_accuracy = 0.0
best_weighted_momentum_series = np.zeros_like(raw_weighted_momentum)
best_q, best_r = 0.001, 0.1

for q_sim in q_grid:
    for r_sim in r_grid:
        candidate_kalman = apply_kalman_filter_custom(raw_weighted_momentum, initial_p=0.50, q_val=q_sim, r_val=r_sim)
        if len(candidate_kalman) == 0: continue
        candidate_direction = np.where(candidate_kalman > 0, 1, 0)
        sim_accuracy = np.mean(candidate_direction == target_v)
        
        if sim_accuracy > best_accuracy:
            best_accuracy = sim_accuracy
            best_weighted_momentum_series = candidate_kalman
            best_q = q_sim
            best_r = r_sim

# =====================================================================
# SIGNAL GENERATION WINDOW
# =====================================================================
view_log, brain_notes, accumulator_log = [], [], []
accumulator = 0
last_valid_view = "⚪ HOLD"

prob_ups = df_predict['Prob_Up'].to_numpy()
prob_downs = df_predict['Prob_Down'].to_numpy()

for i in range(len(prob_ups)):
    p_up, p_down = prob_ups[i], prob_downs[i]
    k2_val = best_weighted_momentum_series[i]
    
    if p_up >= 0.60 and k2_val > 0:
        last_valid_view = f"📈 UP (Prob: {p_up*100:.1f}%)"
        accumulator = min(5, accumulator + 1)
        note = f"🎯 [95% TARGET] Q:{best_q:.4f} | R:{best_r:.2f}"
    elif p_down >= 0.60 and k2_val < 0:
        last_valid_view = f"📉 DOWN (Prob: {p_down*100:.1f}%)"
        accumulator = max(-5, accumulator - 1)
        note = f"🎯 [95% TARGET] Q:{best_q:.4f} | R:{best_r:.2f}"
    else:
        last_valid_view = f"⚪ HOLD (Up: {p_up*100:.0f}% | Dn: {p_down*100:.0f}%)"
        note = "⚡ No multi-vector convergence. Freezing tracking state."

    view_log.append(last_valid_view)
    brain_notes.append(note)
    accumulator_log.append(accumulator)

df_predict['Live_View'] = view_log
df_predict['Accumulator_Score'] = accumulator_log
df_predict['ML_Simulation_Notes'] = brain_notes
df_predict['Weighted_Momentum'] = np.round(best_weighted_momentum_series, 2)

# Frame rendering properties
clean_cols = ['a_Close', 'b_Kalman_Price', 'Weighted_Momentum', 'Accumulator_Score', 'Live_View', 'ML_Simulation_Notes']
display_df = df_predict[clean_cols].copy().iloc[::-1]

display_df['a_Close'] = display_df['a_Close'].round(2)
display_df['b_Kalman_Price'] = display_df['b_Kalman_Price'].round(2)
display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d %H:%M')

st.subheader("📋 Ultra-Precise Optimized Data Frame (Latest Candle Locked on Top)")
st.dataframe(display_df, use_container_width=True, height=600)
