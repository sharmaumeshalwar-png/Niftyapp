import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.title("Nifty 50 Market Regime Data Table")
st.write("1-Hour Candle States (Python 3.14.6 Compatible - No Graph, Pure Data)")

# ==========================================
# STEP 1: DATA INGESTION (Frozen/Local Concept)
# ==========================================
@st.cache_data
def load_frozen_data():
    try:
        ticker = "^NSEI"
        df = yf.download(ticker, start="2025-01-01", interval="1h", auto_adjust=True, threads=False)
        if df.empty:
            raise ValueError()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.to_csv("nifty_frozen.csv")
        return df
    except:
        try:
            return pd.read_csv("nifty_frozen.csv", index_col=0, parse_dates=True)
        except:
            dates = pd.date_range(start="2025-01-01", periods=500, freq="h")
            return pd.DataFrame({"Close": np.sin(np.linspace(0, 20, 500)) * 500 + 23000, 
                                 "High": 23500, "Low": 22500}, index=dates)

data = load_frozen_data()

# ==========================================
# STEP 2 & 3: FEATURE ENGINEERING
# ==========================================
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=10).std()
data.dropna(inplace=True)

# ==========================================
# STEP 4 & 5: REGIME SWITCHING LOGIC
# ==========================================
def calculate_regimes(df):
    states = []
    v_mean = df['Volatility'].mean()
    v_std = df['Volatility'].std()
    
    for idx, row in df.iterrows():
        if row['Volatility'] > (v_mean + 0.5 * v_std) and row['Returns'] < 0:
            states.append("State 1 (Bearish Downtrend)")
        elif row['Volatility'] < v_mean and row['Returns'] > -0.001:
            states.append("State 0 (Bullish Momentum)")
        else:
            states.append("State 2 (High Volatility/Sideways)")
            
    return states

data['Market_Regime'] = calculate_regimes(data)

# ==========================================
# STEP 6 & 7: DATA COMPLIANCE & TABLE FORMATTING
# ==========================================
# Table ko readable banane ke liye columns clean kar rahe hain
final_table = data[['Open', 'High', 'Low', 'Close', 'Returns', 'Market_Regime']].copy()

# Formatting percentages for better readability
final_table['Returns'] = final_table['Returns'].apply(lambda x: f"{x*100:.2f}%")

# Latest candles sabse upar dikhane ke liye reverse chronological order
final_table = final_table.sort_index(ascending=False)

# Displaying the main data table in Streamlit
st.subheader("📋 Nifty 50 Hourly Regime Log (Latest First)")
st.dataframe(final_table, use_container_width=True)

# ==========================================
# STEP 8: STATISTICAL ANALYSIS SUMMARY TABLE
# ==========================================
st.subheader("📊 Summary Statistics Table")

summary_data = []
labels = ["State 0 (Bullish Momentum)", "State 1 (Bearish Downtrend)", "State 2 (High Volatility/Sideways)"]

for label in labels:
    state_data = data[data['Market_Regime'] == label]
    summary_data.append({
        "Market Regime": label,
        "Total 1-Hr Candles": len(state_data),
        "Avg Hourly Return": f"{state_data['Returns'].mean()*100:.4f}%",
        "Risk/Volatility": f"{state_data['Volatility'].mean()*100:.4f}%"
    })

summary_df = pd.DataFrame(summary_data)
st.table(summary_df)
