import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: DATA INGESTION (Nifty 50)
# ==========================================
print("Step 1: Fetching Nifty 50 Data...")
# Nifty 50 Index ticker in Yahoo Finance is '^NSEI'
ticker = "^NSEI"
data = yf.download(ticker, start="2024-01-01", end="2026-06-25")

# ==========================================
# STEP 2: FEATURE ENGINEERING
# ==========================================
print("Step 2: Calculating Returns and Volatility...")
data['Returns'] = data['Adj Close'].pct_change()
data['Range'] = (data['High'] - data['Low']) / data['Close']
data.dropna(inplace=True)

# Preparing features for HMM (Returns and Intra-day Range)
X = data[['Returns', 'Range']].values

# ==========================================
# STEP 3: MODEL CONFIGURATION (HMM)
# ==========================================
print("Step 3: Configuring Hidden Markov Model...")
# Hum 3 Hidden States assume kar rahe hain: Bull, Bear, Sideways
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)

# ==========================================
# STEP 4: MODEL TRAINING (FITTING)
# ==========================================
print("Step 4: Training the Model...")
model.fit(X)

# ==========================================
# STEP 5: STATE PREDICTION
# ==========================================
print("Step 5: Predicting Hidden States...")
hidden_states = model.predict(X)
data['State'] = hidden_states

# ==========================================
# STEP 6: POST-PROCESSING & REGIME LABELLING
# ==========================================
print("Step 6: Analyzing State Characteristics...")
for i in range(model.n_components):
    state_data = data[data['State'] == i]
    print(f"State {i} -> Mean Return: {state_data['Returns'].mean():.4f}, Volatility: {state_data['Returns'].std():.4f}")

# ==========================================
# STEP 7: PLOTTING & VISUALIZATION
# ==========================================
print("Step 7: Plotting Nifty Regimes...")
plt.figure(figsize=(15, 8))

# Define colors for 3 different states
colors = {0: 'green', 1: 'red', 2: 'blue'}
labels = {0: 'State 0 (Growth/Bull)', 1: 'State 1 (High Volatility/Bear)', 2: 'State 2 (Consolidation)'}

# Plotting the Close price with state colors
for i in range(len(data) - 1):
    plt.plot(data.index[i:i+2], data['Adj Close'].iloc[i:i+2], 
             color=colors[data['State'].iloc[i]], linewidth=2)

# Custom Legend Creating
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(3)]
plt.legend(handles=legend_elements, loc='upper left')

plt.title('Nifty 50 Market Regimes Predicted by Hidden Markov Model (HMM)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Nifty Close Price', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# STEP 8: OUTCOME GENERATION
# ==========================================
print("\nStep 8: [SUCCESS] Execution Complete. Plot generated successfully.")
