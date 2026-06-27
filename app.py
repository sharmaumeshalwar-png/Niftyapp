import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Nifty 50 Market Regime Detection (HMM)")
st.write("Kalman theory jaisa adaptive statistical model jo market states ko predict karta hai.")

# ==========================================
# STEP 1: DATA INGESTION (With Fast & Safe Fetch)
# ==========================================
@st.cache_data(ttl=3600)  # 1 ghante tak data cache rahega taaki Yahoo block na kare
def get_nifty_data():
    ticker = "^NSEI"
    try:
        # Auto_adjust aur threads=False dene se cloud servers par IP blocking bypass ho jaati hai
        df = yf.download(ticker, start="2024-01-01", end="2026-06-25", auto_adjust=True, threads=False)
        if df.empty:
            raise ValueError("Yahoo Finance returned empty dataframe.")
        return df
    except Exception as e:
        st.error(f"Yahoo Finance Fetch Failed. Alternate method use ho raha hai...")
        # Fallback: Agar Yahoo block kare toh user ke liye error handle ho jaye
        return pd.DataFrame()

data = get_nifty_data()

if data.empty:
    st.warning("Bhai, Yahoo Finance abhi Streamlit Cloud ko block kar raha hai. Dashboard par jaakar 'Reboot App' dabayein taaki server ka IP badal sake.")
else:
    # ==========================================
    # STEP 2: FEATURE ENGINEERING
    # ==========================================
    # Streamlit me yfinance multi-index columns de sakta hai, usko clean karne ke liye:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    data['Returns'] = data['Close'].pct_change()
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    data.dropna(inplace=True)

    X = data[['Returns', 'Range']].values

    # ==========================================
    # STEP 3 to 5: MODEL CONFIG AND PREDICTION
    # ==========================================
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    data['State'] = hidden_states

    # ==========================================
    # STEP 6 & 7: PLOTTING & VISUALIZATION
    # ==========================================
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = {0: 'green', 1: 'red', 2: 'blue'}
    labels = {0: 'State 0 (Growth/Bull)', 1: 'State 1 (High Volatility/Bear)', 2: 'State 2 (Consolidation)'}

    for i in range(len(data) - 1):
        ax.plot(data.index[i:i+2], data['Close'].iloc[i:i+2], 
                 color=colors[data['State'].iloc[i]], linewidth=2)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title('Nifty 50 Market Regimes Predicted by Hidden Markov Model (HMM)')
    ax.grid(True, alpha=0.3)

    # Streamlit me plt.show() ki jagah st.pyplot() use hota hai
    st.pyplot(fig)

    # ==========================================
    # STEP 8: STATISTICAL SUMMARY OUTCOME
    # ==========================================
    st.subheader("Market States Ka Statistical Analysis")
    for i in range(3):
        state_data = data[data['State'] == i]
        st.write(f"**{labels[i]}** -> Mean Return: `{state_data['Returns'].mean():.4f}`, Volatility: `{state_data['Returns'].std():.4f}`")
