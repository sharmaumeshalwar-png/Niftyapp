# ==========================================
# 1. RAW DATA BREAKDOWN (Before Freeze Mask)
# ==========================================
# Lookback space leakage ko prevent karne ke liye calculation hamesha raw dataframe par hogi
raw_df['ColumnC_Delta'] = raw_df['Close'] - raw_df['Close'].shift(int(delta_lookback))

# Coppock Engine calculations on unmasked raw series
raw_df['ROC_Long'] = ((raw_df['Close'] - raw_df['Close'].shift(int(long_roc))) / raw_df['Close'].shift(int(long_roc))) * 100
raw_df['ROC_Short'] = ((raw_df['Close'] - raw_df['Close'].shift(int(short_roc))) / raw_df['Close'].shift(int(short_roc))) * 100
raw_df['RawMatrixSum'] = raw_df['ROC_Long'] + raw_df['ROC_Short']

# WMA Smoothing array configuration
weights = np.arange(1, int(wma_smoothing) + 1)
raw_df['CoppockCurve'] = raw_df['RawMatrixSum'].rolling(int(wma_smoothing)).apply(
    lambda x: np.dot(x, weights) / weights.sum(), raw=True
)

# ==========================================
# 2. SIGNAL VERIFICATION MATRIX (Step-by-Step Loop)
# ==========================================
# Shift variables to identify accurate crossover timestamps
raw_df['Coppock_Prev'] = raw_df['CoppockCurve'].shift(1)

raw_df['Bullish_Hint'] = (raw_df['ColumnC_Delta'] > 0) & (raw_df['CoppockCurve'] > 0) & (raw_df['Coppock_Prev'] <= 0)
raw_df['Bearish_Hint'] = (raw_df['ColumnC_Delta'] < 0) & (raw_df['CoppockCurve'] < 0) & (raw_df['Coppock_Prev'] >= 0)

# ==========================================
# 3. STRICT 2-YEAR FREEZE & OUTCOME MASK
# ==========================================
# Calculations hone ke baad hi data ko freeze coordinates par trim kiya jayega
start_freeze = pd.Timestamp("2025-01-01 00:00:00")
end_freeze = pd.Timestamp("2026-12-31 23:59:59")

# Final isolated matrix that guarantees zero lookback leakage
df = raw_df[(raw_df.index >= start_freeze) & (raw_df.index <= end_freeze)].copy()
