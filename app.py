import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit Page Configuration
st.set_page_config(page_title="Kalman + ML Hybrid Predictor", layout="wide")
st.title("📈 Kalman Filter & Machine Learning Hybrid Model")
st.write("A = Close Price ➡️ B = Kalman Filter (Q=0.0001) ➡️ C = Features (A & B) ➡️ D = ML Prediction")

# -------------------------------------------------------------------------
# Helper Function: 1D Kalman Filter Implementation
# -------------------------------------------------------------------------
def apply_kalman_filter(prices, Q=0.0001, R=0.1):
    """
    Applies a simple 1D Kalman Filter to a series of prices.
    Q: Process variance (given as 0.0001 by user)
    R: Measurement variance (assumed noise in market data)
    """
    n_timestamps = len(prices)
    filtered_prices = np.zeros(n_timestamps)
    
    # Initial guesses
    x_hat = prices[0]  # initial state estimate
    P = 1.0            # initial error covariance
    
    for t in range(n_timestamps):
        # 1. Prediction Step
        x_hat_minus = x_hat
        P_minus = P + Q
        
        # 2. Measurement Update Step (Correction)
        K = P_minus / (P_minus + R)  # Kalman Gain
        x_hat = x_hat_minus + K * (prices[t] - x_hat_minus)
        P = (1 - K) * P_minus
        
        filtered_prices[t] = x_hat
        
    return filtered_prices

# -------------------------------------------------------------------------
# Sidebar: Data Input Setup
# -------------------------------------------------------------------------
st.sidebar.header("📊 Data & Parameter Settings")
data_option = st.sidebar.selectbox("Data Source Select Karein", ["Generate Synthetic Data", "Upload CSV"])

if data_option == "Generate Synthetic Data":
    st.sidebar.subheader("Synthetic Data Parameters")
    n_days = st.sidebar.slider("Kitne din ka data?", 100, 1000, 5000)
    
    # Generate random-walk style close prices (A)
    np.random.seed(42)
    steps = np.random.normal(0, 0.5, n_days)
    close_prices = 100 + np.cumsum(steps)
    # Adding some noise
    close_prices += np.random.normal(0, 0.2, n_days)
    
    df = pd.DataFrame({"Close_A": close_prices})
else:
    uploaded_file = st.sidebar.file_uploader("CSV file upload karein (Isme 'Close' column hona chahiye)", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        if 'Close' in user_df.columns:
            df = pd.DataFrame({"Close_A": user_df['Close'].values})
        else:
            st.error("Error: CSV file mein 'Close' naam ka column nahi mila!")
            st.stop()
    else:
        st.info("Aapki kisi CSV file ka intezar hai. Tab tak dummy data dekhne ke liye sidebar se 'Generate Synthetic Data' chunein.")
        st.stop()

# -------------------------------------------------------------------------
# Step-by-Step Processing Framework (8-Step Logic Integration)
# -------------------------------------------------------------------------

# Step 1 & 2: Get A and Compute B (Kalman Filter)
df['Kalman_B'] = apply_kalman_filter(df['Close_A'].values, Q=0.0001, R=0.1)

# Step 3: Combine into C (Features) and Shift for Next-Day Target Prediction
df['Feature_A_Lag'] = df['Close_A']
df['Feature_B_Lag'] = df['Kalman_B']
# Target Variable D: Next day's close price
df['Target_D'] = df['Close_A'].shift(-1) 

# Drop the last row because it won't have a target value
df = df.dropna()

# Step 4 & 5: Train-Test Split (80% Train, 20% Test for validation)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[['Feature_A_Lag', 'Feature_B_Lag']]
y_train = train_df['Target_D']
X_test = test_df[['Feature_A_Lag', 'Feature_B_Lag']]
y_test = test_df['Target_D']

# Step 6: Machine Learning (D) Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Prediction Phase
test_df = test_df.copy()
test_df['Predicted_D'] = model.predict(X_test)

# Step 8: Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test, test_df['Predicted_D']))
r2 = r2_score(y_test, test_df['Predicted_D'])

# -------------------------------------------------------------------------
# Streamlit UI Dashboard Elements
# -------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric(label="📊 Test Data RMSE (Error)", value=f"{rmse:.4f}")
with col2:
    st.metric(label="🎯 Model R² Score (Accuracy)", value=f"{r2*100:.2f}%")

st.markdown("---")
st.subheader("📈 Visualization Dashboard")

# Plotting the Results
fig, ax = plt.subplots(figsize=(14, 7))

# Plotting a subset of test data for clear visibility
plot_len = min(150, len(test_df))
plot_df = test_df.tail(plot_len).reset_index()

ax.plot(plot_df['Close_A'], label="A: Actual Close Price", color="black", alpha=0.4, linestyle="--")
ax.plot(plot_df['Kalman_B'], label="B: Kalman Filtered (Q=0.0001)", color="blue", linewidth=1.5)
ax.plot(plot_df['Predicted_D'], label="D: ML Hybrid Prediction (Next Step)", color="red", linewidth=2)

ax.set_title(f"Last {plot_len} Timestamps Comparison")
ax.set_xlabel("Timestamps")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Show raw processed dataset matrix C
st.subheader("📋 Processed Data Matrix (C)")
st.dataframe(df[['Close_A', 'Kalman_B', 'Target_D']].tail(10))
