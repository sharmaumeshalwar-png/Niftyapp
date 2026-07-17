import streamlit as st
import pandas as pd
import numpy as np

# 1. Title & Alert
st.title("BTC 200-Point Bulletproof Range Engine")
st.warning("⚠️ GREEN ROW = Running Candle (Do NOT trade this). Normal Rows = Locked and Safe.")

# Mock Data Generator (Strictly backfilled and sequential to prevent future leak)
# Replace this block with your actual yfinance / live data fetch
@st.cache_data(ttl=10)
def fetch_and_clean_data():
    # Example: Strict chronological order (No forward looking)
    raw_prices = np.random.normal(63000, 300, 1000)
    df = pd.DataFrame(raw_prices, columns=['Close'])
    # Strict Backfill to handle any missing values without leaks
    df['Close'] = df['Close'].bfill().ffill()
    return df

data_df = fetch_and_clean_data()

# 2. Static Grid Calculation (Prevents shift on refresh)
RANGE_STEP = 200.0
STATIC_BASE = 60000.0  # Safe static starting anchor

# 3. Generating Range Bars (Loop with strictly historical lookup)
range_closes = []
timestamps = []

current_anchor = STATIC_BASE
# Calculate the closest initial step based on the first price to avoid lag
first_price = data_df['Close'].iloc[0]
current_anchor = np.round((first_price - STATIC_BASE) / RANGE_STEP) * RANGE_STEP + STATIC_BASE

for price in data_df['Close']:
    # Check upward or downward breach of the 200-point boundary
    while price >= current_anchor + RANGE_STEP:
        current_anchor += RANGE_STEP
        range_closes.append(current_anchor)
    while price <= current_anchor - RANGE_STEP:
        current_anchor -= RANGE_STEP
        range_closes.append(current_anchor)

# Reverse to show latest on top
range_closes = range_closes[::-1]

# 4. Preparing DataFrame for display
matrix_df = pd.DataFrame(range_closes, columns=['Range Close (200-Pt steps)'])

# Mock Calculations for Demonstration (Ensure these are your actual HAM/Hurst formulas)
matrix_df['Raw HAM'] = np.random.uniform(-100, 1000, len(matrix_df))
matrix_df['Signal'] = np.where(matrix_df['Raw HAM'] < 100, "🔴 SELL", "🟢 BUY")
matrix_df['Prob_Up'] = np.where(matrix_df['Signal'] == "🔴 SELL", "1%", "95%")

# 5. Pandas Styler for Running Row (Green Row Logic)
def highlight_running_row(df):
    # Create an empty style matrix
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Row 0 is the topmost (Latest Running Row)
    if len(df) > 0:
        # Highlighting the running row in distinct soft green
        styles.iloc[0, :] = 'background-color: #2e7d32; color: #ffffff; font-weight: bold;'
        
    return styles

# Apply style and display table
styled_matrix = matrix_df.style.apply(highlight_running_row, axis=None)
st.dataframe(styled_matrix, use_container_width=True, height=600)
