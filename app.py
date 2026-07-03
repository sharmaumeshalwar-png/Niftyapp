import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Bitcoin Ultra-Responsive Engine", layout="wide")
st.title("⚡ Bitcoin (BTC) Live Dynamic-Flip Permanent Engine")
st.write("🎯 **Aapki Perfect Permanent Setting:** 13-Day Rolling Anchor (Saare Hints Restored & Locked)")

# =====================================================================
# MATHEMATICAL ENGINE (Kalman Filter 0.001)
# =====================================================================
def apply_kalman_filter_strict(price_array):
    if len(price_array) == 0:
        return []
    x = price_array[0]
    p = 50.0  
    q = 0.001  
    r = 0.1    
    filtered_prices = []
    for z in price_array:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered_prices.append(x)
    return filtered_prices

with st.spinner("Aligning Responsive Crypto Microstructure Matrices..."):
    # Fix 1: Period ko 30d kiya taaki 5m interval safely download ho sake bina timeout ke
    df = yf.download("BTC-USD", period="30d", interval="5m")
    
    # Fix 2: Ultra-Safe MultiIndex Column Flattening
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = [col[0] for col in df.columns]

    if df.empty or len(df) < 50:
        st.error("⚠️ Data download nahi ho paya ya data bohot kam hai. Please page ko refresh karein!")
        st.stop()

    df.index = pd.to_datetime(df.index)

    # Base Matrix Definition
    df['a_Close'] = df['Close']
    df['b_Kalman'] = apply_kalman_filter_strict(df['a_Close'].values)
    df['c_Combined'] = df['a_Close'] - df['b_Kalman']
    
    df['Sign_Change'] = np.sign(df['c_Combined']) != np.sign(df['c_Combined'].shift(1))
    df['Sign_Change'] = df['Sign_Change'].astype(int)
    
    # Microstructure Features
    df['Order_Imbalance'] = (df['a_Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Body_Center'] = (df['Open'] + df['a_Close']) / 2
    df['Body_Imbalance'] = (df['Body_Center'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    rolling_std = df['c_Combined'].rolling(window=24).std() + 1e-10
    df['Normalized_Gap'] = df['c_Combined'] / rolling_std
    df['Flow_Velocity'] = df['c_Combined'].diff(1)
    
    # Target Configuration (3 periods forward shift)
    df['Target'] = np.where(df['a_Close'].shift(-3) > df['a_Close'], 1, 0)
    
    # Fix 3: Srf prediction features se naff values clear karna, full row optimization ke saath
    features_matrix = ['c_Combined', 'Order_Imbalance', 'Body_Imbalance', 'Normalized_Gap', 'Flow_Velocity']
    df.dropna(subset=features_matrix, inplace=True)

# =====================================================================
# 🌟 THE GOLDEN SOLUTION: 13-DAY AUTOMATIC ROLLING BOUNDARY ANCHOR
# =====================================================================
start_data_date = df.index.min()
split_boundary_date = start_data_date + pd.Timedelta(days=13)  # 🔴 STRICTLY 13 DAYS FIXED FOR ANCHOR RESET

train_mask = df.index < split_boundary_date
predict_mask = df.index >= split_boundary_date

df_train = df[train_mask].dropna(subset=['Target'])
X_train = df_train[features_matrix]
y_train = df_train['Target']
X_predict = df.loc[predict_mask, features_matrix]

if len(X_predict) == 0 or len(X_train) == 0:
    st.warning(f"Data split alignment warning: Training rows ({len(X_train)}) | Prediction rows ({len(X_predict)}). Data size kam hai, isliye automatic adjustment operational mode apply ho rha h.")
    # Fallback: agar 13 days ke baad data na bache toh 50-50 split use karega taaki blank na ho screen
    midpoint = len(df) // 2
    X_train = df[features_matrix].iloc[:midpoint]
    y_train = df['Target'].iloc[:midpoint]
    X_predict = df[features_matrix].iloc[midpoint:]
    df_signals = df.iloc[midpoint:].copy()
else:
    df_signals = df[predict_mask].copy()

# Model Execution Guard
if len(X_train) > 0 and len(X_predict) > 0:
    # 🔴 AAPKI PERFECT LOW SETTING FOR FAST DIFFERENTIATION
    model_flow = RandomForestClassifier(
        n_estimators=150, 
        max_depth=3,            
        min_samples_leaf=1,     
        random_state=42
    )
    model_flow.fit(X_train, y_train)

    probabilities = model_flow.predict_proba(X_predict)
    
    df_signals['Prob_Down'] = probabilities[:, 0]
    df_signals['Prob_Up'] = probabilities[:, 1]

    # =====================================================================
    # LIVE DYNAMIC AUTO-FLIP CIRCUIT 🛡️
    # =====================================================================
    final_signals = []
    current_state = "HOLD"

    sign_changes = df_signals['Sign_Change'].values
    prob_ups = df_signals['Prob_Up'].values
    prob_downs = df_signals['Prob_Down'].values

    for i in range(len(df_signals)):
        sc = sign_changes[i]
        p_up = prob_ups[i]
        p_down = prob_downs[i]

        # 1. Fresh Signal Rule via Kalman Cross
        if sc == 1:
            if p_up >= 0.60:  
                current_state = "BUY"
                final_signals.append("🟢 INSTITUTIONAL BUY (Confirmed)")
            elif p_down >= 0.60:
                current_state = "SELL"
                final_signals.append("🔴 INSTITUTIONAL SELL (Confirmed)")
            else:
                current_state = "HOLD"
                final_signals.append("⚪ HOLD")
        
        # 2. Continuous Monitoring (The Auto-Flip / Emergency Circuit)
        else:
            if current_state == "BUY" and p_down >= 0.65:
                current_state = "SELL"
                final_signals.append("🔴 AUTO-FLIP TO SELL (Risk Shield)")
            elif current_state == "SELL" and p_up >= 0.65:
                current_state = "BUY"
                final_signals.append("🟢 AUTO-FLIP TO BUY (Risk Shield)")
            else:
                if current_state == "BUY":
                    final_signals.append("🟢 HOLD BUY")
                elif current_state == "SELL":
                    final_signals.append("🔴 HOLD SELL")
                else:
                    final_signals.append("⚪ HOLD")

    df_signals['Engine_Signal'] = final_signals

    # =====================================================================
    # DASHBOARD OUTPUT DISPLAY
    # =====================================================================
    st.subheader("📊 Engine Signal Live Tracking Matrix")
    st.dataframe(df_signals[['a_Close', 'b_Kalman', 'c_Combined', 'Prob_Up', 'Prob_Down', 'Engine_Signal']].tail(50))
else:
    st.error("Data crunching state failed. Data processing pipeline clear karke wapas run karein.")
