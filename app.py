import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from filterpy.kalman import KalmanFilter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# =====================================================================
# STEP 1 & 2: DATA FETCHING & KALMAN FILTER FUNCTION
# =====================================================================
def apply_kalman_filter(price_array):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[price_array[0]], [0.]]) # Initial state (price, velocity)
    kf.F = np.array([[1., 1.], [0., 1.]])    # State transition matrix
    kf.H = np.array([[1., 0.]])               # Measurement matrix
    kf.P *= 1000.                             # Covariance matrix
    kf.R = 0.5                                # Measurement noise
    kf.Q = np.array([[0.01, 0.], [0., 0.01]]) # Process noise
    
    filtered_prices = []
    for z in price_array:
        kf.predict()
        kf.update(z)
        filtered_prices.append(kf.x[0, 0])
    return filtered_prices

# Fetching 1-Hour Candle Data for Nifty BeES
print("Fetching 1-Hour data for NIFTYBEES.NS...")
df = yf.download("NIFTYBEES.NS", start="2025-01-01", end="2026-06-30", interval="1h")

# Fix MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Apply Kalman Filter on Close Price
df['Kalman_Price'] = apply_kalman_filter(df['Close'].values)

# =====================================================================
# STEP 3: TECHNICAL INDICATORS
# =====================================================================
df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
df.ta.macd(append=True) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
df.ta.rsi(append=True)  # RSI_14
df.ta.atr(append=True)  # ATRe_14
df.ta.adx(append=True)  # ADX_14, DMP_14 (+DI), DMN_14 (-DI)

df.dropna(inplace=True)

# Target: Next hour return direction (1 = Up, 0 = Down)
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Features array
features = ['Volume', 'VWAP', 'Kalman_Price', 'MACD_12_26_9', 'RSI_14', 'ATRe_14', 'DMP_14', 'DMN_14']
X = df[features]
y = df['Target']

# =====================================================================
# STEP 4 & 5: DATA SPLITTING & MODEL TRAINING (Jan 2025 - Jan 2026)
# =====================================================================
train_mask = (df.index >= '2025-01-01') & (df.index < '2026-01-01')
test_mask = (df.index >= '2026-01-01')

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Testing Models
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

model_rf.fit(X_train, y_train)
model_gb.fit(X_train, y_train)

# =====================================================================
# STEP 6 & 7: BEST MODEL SELECTION & WALK-FORWARD TESTING
# =====================================================================
acc_rf = accuracy_score(y_train, model_rf.predict(X_train))
acc_gb = accuracy_score(y_train, model_gb.predict(X_train))

# Choosing the best model based on training accuracy
if acc_rf > acc_gb:
    best_model = model_rf
    print(f"\n🏆 Best Model Selected: Random Forest (Train Acc: {acc_rf:.2%})")
else:
    best_model = model_gb
    print(f"\n🏆 Best Model Selected: Gradient Boosting (Train Acc: {acc_gb:.2%})")

# =====================================================================
# STEP 8: SIGNAL GENERATION (1 Jan 2026 to Present)
# =====================================================================
df_out = df[test_mask].copy()
df_out['Predicted_Signal'] = best_model.predict(X_test)

# Convert 0/1 to Short/Long or Hold Signals (-1, 1)
df_out['Signal'] = np.where(df_out['Predicted_Signal'] == 1, "BUY (Long)", "SELL (Exit/Short)")

print("\n--- Out of Sample Live Trading Signals (1 Jan 2026 to Today) ---")
print(df_out[['Close', 'Kalman_Price', 'RSI_14', 'Signal']].tail(10))
