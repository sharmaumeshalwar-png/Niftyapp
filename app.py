import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Kalman + ML Matrix", layout="wide")
st.title("📊 Nifty 50: Kalman Filter & Machine Learning Hybrid Matrix")
st.write("A = Nifty Close Price | B = Kalman Filter (Q=0.0001) | C = Features | D = ML Next Hour Prediction")
st.write("**Data Source:** Yahoo Finance Live (`^NSEI`) | **Interval:** 1-Hour | **Start Date:** 01 Jan 2025")

# -------------------------------------------------------------------------
# Kalman Filter Function
# -------------------------------------------------------------------------
def apply_kalman_filter(prices, Q=0.0001, R=0.1):
    n_timestamps = len(prices)
    filtered_prices = np.zeros(n_timestamps)
    
    x_hat = prices[0]  
    P = 1.0            
    
    for t in range(n_timestamps):
        x_hat_minus = x_hat
        P_minus = P + Q
        
        K = P_minus / (P_minus + R)  
        x_hat = x_hat_minus + K * (prices[t] - x_hat_minus)
        P = (1 - K) * P_minus
        
        filtered_prices[t] = x_hat
        
    return filtered_prices

# -------------------------------------------------------------------------
# Live Nifty Data Fetching (1 Jan 2025 se aaj tak)
# -------------------------------------------------------------------------
@st.cache_data # Data ko baar-baar download hone se bachane ke liye caching
def load_nifty_data():
    # Nifty 50 ka ticker ^NSEI hai
    ticker = "^NSEI"
    # 1 Jan 2025 se data fetch karna
    data = yf.download(ticker, start="2025-01-01", interval="1h")
    
    # Agar multi-index columns hain toh unhe clean karna
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    return data

try:
    with st.spinner("Yahoo Finance se Nifty ka 1-Hour data load ho raha hai..."):
        nifty_raw = load_nifty_data()

    if nifty_raw.empty:
        st.error("Nifty data fetch nahi ho paya. Internet connection check karein.")
        st.stop()
        
    # Raw close price (A) extraction
    df = pd.DataFrame({"Close_A": nifty_raw['Close'].dropna()})

    # -------------------------------------------------------------------------
    # Processing Framework (Kalman + ML)
    # -------------------------------------------------------------------------
    # Step 1 & 2: Calculate Kalman Filter (B) on Nifty Close (A)
    df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.1)

    # Step 3: Feature Matrix C (Lags)
    df['Feature_A_Lag'] = df['Close_A']
    df['Feature_B_Lag'] = df['Kalman_B']

    # Target Variable D: Next Hour's Nifty Close Price
    df['Target_D_Next_Hour'] = df['Close_A'].shift(-1)
    df_clean = df.dropna()

    # Step 4 & 5: Train-Test Split (80% Train / 20% Test)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    X_train = train_df[['Feature_A_Lag', 'Feature_B_Lag']]
    y_train = train_df['Target_D_Next_Hour']
    X_test = test_df[['Feature_A_Lag', 'Feature_B_Lag']]
    y_test = test_df['Target_D_Next_Hour']

    # Step 6: ML Training
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Step 7: Prediction
    test_df = test_df.copy()
    test_df['ML_Prediction_D'] = model.predict(X_test)

    # -------------------------------------------------------------------------
    # Output Matrix Table Display
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Nifty 50 Output Data Matrix Table")
    
    # Table formatting
    output_table = test_df[['Close_A', 'Kalman_B', 'Target_D_Next_Hour', 'ML_Prediction_D']].copy()
    output_table.columns = [
        "Nifty Actual Close (A)", 
        "Kalman Smooth (B)", 
        "Next Hour Target (True D)", 
        "ML Prediction (Predicted D)"
    ]
    
    # Formatting indices and resetting for clean view
    output_table.index = output_table.index.strftime('%Y-%m-%d %H:%M')
    output_table = output_table.reset_index()
    output_table.rename(columns={'index': 'Date & Time (Hourly)'}, inplace=True)

    # Dynamic Rows Slider
    rows_to_show = st.slider("Table mein kitne rows dekhne hain?", 10, 200, 20)
    
    # Pure HTML standard Table grid (Blank view nahi aayega)
    st.table(output_table.tail(rows_to_show))

except Exception as e:
    st.error(f"Kuch error aaya hai: {e}")
