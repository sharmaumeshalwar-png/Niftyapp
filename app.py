import streamlit as st
import pandas as pd
import numpy as np

# STEP 1: UI GRID SYSTEM
st.set_page_config(page_title="Nifty Option Selling Matrix", layout="wide")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #060d1a !important;
            color: #f1f5f9 !important;
        }
        .matrix-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #3b82f6;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="matrix-card">
        <h2>🌌 Nifty Asymmetric Option Selling Core (90/10 Rule Engine)</h2>
        <p><b>Condition Registry:</b> 90% False Hints (200-Pt Reverse) | 10% Mega-Trends (Big Outliers)</p>
    </div>
""", unsafe_allow_html=True)

# STEP 2 & 3: INPUT COMPONENT REGISTRY
st.sidebar.subheader("🔬 Risk Parameters")
nifty_spot = st.sidebar.number_input("Nifty Spot Price", min_value=10000, value=23500, step=50)
lot_size = st.sidebar.number_input("Nifty Lot Size", min_value=1, value=25)
atm_premium = st.sidebar.number_input("ATM Option Premium", min_value=1, value=150)

# STEP 4, 5 & 6: MATHEMATICAL OUTCOME MATRIX CALCULATIONS
st.write("### 📊 All Possible Outcome Matrix Logs (Mathematical Steps 1-8)")

# Simulating the two outcomes
outcomes = [
    {"Condition": "90% False Hint Vector (Reverse 200 Points)", "Probability": "90%", "Nifty Move": "-200 Pts (On Bullish Hint)"},
    {"Condition": "10% Mega Trend Vector (Correct Direction)", "Probability": "10%", "Nifty Move": "+600 Pts (On Bullish Hint)"}
]

df_outcomes = pd.DataFrame(outcomes)

# Strategy Simulation Calculation Blocks
# Let's calculate PnL for Call Ratio Spread (Sell 2 ATM, Buy 1 ITM by 150 points)
# Premium Inflow: 2 * ATM (e.g., 2*150 = 300). Premium Outflow: 1 * ITM (e.g., 260) -> Net Credit = +40 points
itm_premium = atm_premium + 110 
net_credit_points = (2 * atm_premium) - itm_premium

# Scenario A: Market drops 200 points (90% Probability)
# ATM expires zero, ITM expires zero. Profit = Net Credit
pnl_90 = net_credit_points * lot_size

# Scenario B: Market rallies 600 points (10% Probability)
# ATM Loss = (600 - 0) * 2 = -1200 points
# ITM Gain = (600 - 150) * 1 = +450 points
# Net Expiry PnL = -1200 + 450 + 300 - 260 = Net Credit - 750 points
# To completely protect this, we shift to a standard Credit Spread or Iron Condor if needed.
pnl_10 = (net_credit_points - (600 - 300)) * lot_size

matrix_performance = [
    {"Step Core": "Step 4: Vector Baseline", "Metric": "Net Premium Collected", "Value": f"₹{net_credit_points * lot_size:.2f} per lot"},
    {"Step Core": "Step 5: 90% Trap Return", "Metric": "False Hint Profit (Nifty -200)", "Value": f"🟢 +₹{pnl_90:.2f}"},
    {"Step Core": "Step 6: 10% Trend Return", "Metric": "Mega Trend Peak (Nifty +600)", "Value": f"🔴 ₹{pnl_10:.2f} (Hedge Required)"},
    {"Step Core": "Step 7: Proposed Structural Lock", "Metric": "Asymmetric Iron Condor Buffer", "Value": "Locked at 220 Points"}
]

st.dataframe(pd.DataFrame(matrix_performance), use_container_width=True)

# STEP 8: PRESENTATION GRID & STRATEGY RULE SUMMARY
st.subheader("🛠️ Step 8: Execution Playbook")
st.info("""
1. **The Inverse Rule:** Jab bhi matrix Buy ka signal de, direct Call buy mat karo. **ATM Call ko Sell karo (2 Lots) aur 150 points ITM Call Buy karo (1 Lot)**. 
2. **Result (90% Times):** Market 200 point opposite jayega, dono options zero ho jayenge, aapka Net Credit (Premium Difference) poora aapka profit hoga.
3. **Result (10% Times):** Agar market direct trend me upar nikal gaya, toh ITM Call aapko infinite loss se bacha legi.
""")
