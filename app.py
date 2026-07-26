import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==============================================================================
# 1. DATA ACQUISITION (2 Years of 1-Hour BTC Data)
# ==============================================================================
print("Step 1: Downloading 2 years (720 days) of 1h BTC-USD data...")
df = yf.download(tickers="BTC-USD", period="720d", interval="1h")

# MultiIndex handling for latest yfinance output
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[['Close']].copy()
df.rename(columns={'Close': 'close'}, inplace=True)


# ==============================================================================
# 2. FEATURE ENGINEERING (Strictly Causal / Zero Lookahead)
# ==============================================================================

# Feature 1: Close Price
# Raw close is available as df['close']

# Feature 2: Causal Kalman Filter (Measurement Variance = 0.50)
def apply_causal_kalman(series, process_variance=1e-5, measurement_variance=0.50):
    n = len(series)
    kalman_out = np.zeros(n)
    
    x_hat = series.iloc[0]
    P = 1.0
    Q = process_variance
    R = measurement_variance
    
    for i in range(n):
        # Time Update (Predict)
        x_hat_minus = x_hat
        P_minus = P + Q
        
        # Measurement Update (Correct)
        K = P_minus / (P_minus + R)
        x_hat = x_hat_minus + K * (series.iloc[i] - x_hat_minus)
        P = (1 - K) * P_minus
        
        kalman_out[i] = x_hat
        
    return kalman_out

print("Step 2: Calculating Kalman Filter (Measurement Variance: 0.50)...")
df['kalman'] = apply_causal_kalman(df['close'], measurement_variance=0.50)

# Feature 3: Weighted Momentum (Close - Kalman via EWMA)
print("Step 3: Calculating Weighted Momentum (Close - Kalman)...")
raw_momentum = df['close'] - df['kalman']
df['weighted_momentum'] = raw_momentum.ewm(span=14, adjust=False).mean()

# Feature 4: Fast Causal Rolling Hurst Exponent (30 Candle Window)
def fast_hurst_30(series, window=30):
    vals = series.values
    n = len(vals)
    hurst_arr = np.full(n, np.nan)
    lags = np.arange(2, 10)
    log_lags = np.log(lags)
    
    for i in range(window, n):
        sub_win = vals[i-window:i]
        tau = []
        for lag in lags:
            diff = sub_win[lag:] - sub_win[:-lag]
            std_val = np.std(diff)
            tau.append(std_val if std_val > 1e-8 else 1e-8)
        
        # Polyfit slope calculation
        poly = np.polyfit(log_lags, np.log(tau), 1)
        hurst_arr[i] = poly[0]
        
    return hurst_arr

print("Step 4: Calculating Rolling Hurst Exponent (30-candle window)...")
df['hurst'] = fast_hurst_30(df['close'], window=30)

# Feature 5: HAM (Hurst * Weighted Momentum)
print("Step 5: Calculating HAM (Hurst * Weighted Momentum)...")
df['ham'] = df['hurst'] * df['weighted_momentum']


# ==============================================================================
# 3. TARGET CREATION & CLEANUP (Strict Dropna - No BFILL)
# ==============================================================================
# Predict if next 1-hour candle close is higher than current candle close
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# LEAK PREVENTION RULE: Drop NaNs created by rolling windows & forward shift.
# Never use bfill() on targets or feature matrix.
df.dropna(inplace=True)

feature_cols = ['close', 'kalman', 'weighted_momentum', 'hurst', 'ham']
X = df[feature_cols]
y = df['target']


# ==============================================================================
# 4. CHRONOLOGICAL 50:50 TRAIN/TEST SPLIT
# ==============================================================================
split_idx = int(len(df) * 0.50)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("\n" + "="*50)
print("DATA SPLIT SUMMARY (Chronological 50:50)")
print("="*50)
print(f"Total Usable Bars: {len(df)}")
print(f"Train Period (Year 1 / 50%): {df.index[0]} to {df.index[split_idx-1]} ({len(X_train)} samples)")
print(f"Test Period  (Year 2 / 50%): {df.index[split_idx]} to {df.index[-1]} ({len(X_test)} samples)")


# ==============================================================================
# 5. MODEL TRAINING & OUT-OF-SAMPLE PREDICTION
# ==============================================================================
# Fit scaler ONLY on X_train to avoid scaling data leak
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5, 
    random_state=42
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


# ==============================================================================
# 6. EVALUATION
# ==============================================================================
acc = accuracy_score(y_test, y_pred)
print("\n" + "="*50)
print(f"OUT-OF-SAMPLE TEST ACCURACY: {acc * 100:.2f}%")
print("="*50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)
