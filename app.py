# 1. Order Imbalance Factor (Volume/Pressure weight)
# Logic: Agar close low ke paas hai (imbalance < 0.2), toh selling pressure hai.
# Agar close high ke paas hai (imbalance > 0.8), toh buying pressure hai.

# 2. Body Imbalance Factor (Candle Strength)
# Logic: Body Center agar low ke paas hai, toh candle bearish hai.

# 3. Combined Interaction Score (Multiplication)
df['Entry_Quality_Score'] = (
    df['c_Combined'] * 0.5 +                   # Momentum Weight
    (df['Order_Imbalance'] - 0.5) * 100 +      # Price Pressure Weight
    (df['Body_Imbalance'] - 0.5) * 50          # Candle Strength Weight
)

# 4. Final Relation with Close Price (The "Signal" Trigger)
# Agar Score Positive hai, toh bullish bias. Negative hai, toh bearish bias.
df['Trade_Signal'] = np.where(df['Entry_Quality_Score'] > 10, "BUY", 
                     np.where(df['Entry_Quality_Score'] < -10, "SELL", "WAIT"))
