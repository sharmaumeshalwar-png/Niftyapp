import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import matplotlib.pyplot as plt

st.title("Nifty 50 Regime Detection (1-Hour Candles)")
st.write("Data frozen hai taaki live fetch ka error dubara na aaye.")

# ==========================================
# STEP 1 & 2: DATA INGESTION (With Local Freezing)
# ==========================================
@st.cache_data
def load_frozen_data():
    # Humne data ko freeze karne ke liye local try-catch lagaya hai
    try:
        # 1-Hour interval ke liye yfinance maximum 1 ya 2 saal ka data hi deta hai
        # Hum 1 Jan 2025 se aaj tak ka 1h data fetch kar rahe hain
        ticker = "^NSEI"
        df = yf.download(ticker, start="2025-01-01", interval="1h", auto_adjust=True, threads=False)
        
        if df.empty:
            raise ValueError("Live fetch failed, using offline fallback.")
            
        # Multi-index columns ko linear karne ke liye flat cleaning
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        # Data ko local memory cache me freeze kar rahe hain
        df.to_csv("nifty_frozen.csv")
        return df
    except Exception as e:
        # Fallback: Agar Yahoo block kare, toh pehle se saved safe data utha lo
        try:
            return pd.read_csv("nifty_frozen.csv", index_col=0, parse_dates=True)
        except:
            # Agar file bhi na ho, toh crash hone se bachane ke liye dummy structure
            st.error("Data fetch completely blocked. Artificial data load ho raha hai.")
            dates = pd.date_range(start="2025-01-01", periods=500, freq="h")
            return pd.DataFrame({"Close": np.sin(np.linspace(0, 20, 500)) * 500 + 23000, 
                                 "High": 23500, "Low": 22500}, index=dates)

# Data local memory me freeze ho chuka hai
data = load_frozen_data()

# ==========================================
# STEP 3 & 4: FEATURE ENGINEERING
# ==========================================
data['Returns'] = data['Close'].pct_change()
data['Range'] = (data['High'] - data['Low']) / data['Close']
data.dropna(inplace=True)

X = data[['Returns', 'Range']].values

# ==========================================
# STEP 5 & 6: HMM TRAINING & STATE PREDICTION
# ==========================================
# 3 States: Bullish, Bearish, Volatile/Sideways
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
model.fit(X)
hidden_states = model.predict(X)
data['State'] = hidden_states

# ==========================================
# STEP 7: PLOTTING THE 1-HOUR CANDLE REGIMES
# ==========================================
fig, ax = plt.subplots(figsize=(15, 7))
colors = {0: 'green', 1: 'red', 2: 'blue'}
labels = {0: 'State 0 (Bullish Momentum)', 1: 'State 1 (Bearish Downtrend)', 2: 'State 2 (High Volatility)'}

# Plotting state lines
for i in range(len(data) - 1):
    ax.plot(data.index[i:i+2], data['Close'].iloc[i:i+2], 
             color=colors[data['State'].iloc[i]], linewidth=1.5)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(3)]
ax.legend(handles=legend_elements, loc='upper left')
ax.set_title('Nifty 50 1-Hour Chart - Hidden Markov Model States (Since Jan 2025)')
ax.set_xlabel('Timeline')
ax.set_ylabel('Nifty Index Points')
ax.grid(True, alpha=0.2)

st.pyplot(fig)

# ==========================================
# STEP 8: FINAL OUTCOME ANALYSIS
# ==========================================
st.subheader("1-Hour Regimes Breakdown")
for i in range(3):
    state_data = data[data['State'] == i]
    st.write(f"**{labels[i]}** -> Total Candles: `{len(state_data)}` | Average Hourly Move: `{state_data['Returns'].mean()*100:.4f}%`")
