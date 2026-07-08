import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Nifty Loop Backtester", layout="wide")

st.title("📊 Nifty ATM Reverse-Loop Backtesting Dashboard")
st.write("Hum se sell pr entry leni h, pe sell pr exit aur naye entry leni h (Only ATM).")

# 1. Aapka Diya Hua Complete Schedule Data
trade_schedule = [
    (2025, 7, 4, 12, 15, 'CE_SELL'),
    (2025, 7, 4, 15, 15, 'PE_SELL'),
    (2025, 7, 8, 10, 15, 'CE_SELL'),
    (2025, 7, 8, 13, 15, 'PE_SELL'),
    (2025, 7, 8, 14, 15, 'CE_SELL'),
    (2025, 7, 8, 15, 15, 'PE_SELL'),
    (2025, 7, 9, 15, 15, 'CE_SELL'),
    (2025, 8, 18, 10, 15, 'PE_SELL'),
    (2025, 8, 26, 10, 15, 'CE_SELL'),
    (2025, 9, 4, 10, 15, 'PE_SELL'),
    (2025, 9, 4, 15, 15, 'CE_SELL'),
    (2025, 9, 8, 10, 15, 'PE_SELL'),
    (2025, 9, 25, 10, 15, 'CE_SELL'),
    (2025, 10, 6, 10, 15, 'PE_SELL'),
    (2025, 11, 4, 14, 15, 'CE_SELL'),
    (2025, 11, 11, 15, 15, 'PE_SELL'),
    (2025, 11, 25, 15, 15, 'CE_SELL'),
    (2025, 11, 26, 10, 15, 'PE_SELL'),
    (2025, 12, 2, 10, 15, 'CE_SELL'),
    (2025, 12, 5, 12, 15, 'PE_SELL'),
    (2025, 12, 8, 10, 15, 'CE_SELL'),
    (2025, 12, 22, 10, 15, 'PE_SELL'),
    (2025, 12, 24, 9, 15, 'CE_SELL'),
    # 2026 ke trades
    (2026, 1, 1, 11, 15, 'PE_SELL'),
    (2026, 1, 1, 12, 15, 'CE_SELL'),
    (2026, 1, 1, 14, 15, 'PE_SELL'),
    (2026, 1, 1, 15, 15, 'CE_SELL'),
    (2026, 1, 2, 10, 15, 'PE_SELL'),
    (2026, 1, 6, 11, 15, 'CE_SELL'),
    (2026, 2, 3, 10, 15, 'PE_SELL'),
    (2026, 2, 5, 14, 15, 'CE_SELL'),
    (2026, 2, 5, 15, 15, 'PE_SELL'),
    (2026, 2, 6, 10, 15, 'CE_SELL'),
    (2026, 2, 6, 12, 15, 'PE_SELL'),
    (2026, 2, 6, 13, 15, 'CE_SELL'),
    (2026, 2, 6, 14, 15, 'PE_SELL'),
    (2026, 2, 13, 10, 15, 'CE_SELL'),
    (2026, 2, 16, 15, 15, 'PE_SELL'),
    (2026, 2, 19, 12, 15, 'CE_SELL'),
    (2026, 2, 23, 10, 15, 'PE_SELL'),
    (2026, 2, 23, 13, 15, 'CE_SELL'),
    (2026, 2, 23, 15, 15, 'PE_SELL'),
    (2026, 2, 24, 10, 15, 'CE_SELL'),
    (2026, 4, 8, 10, 15, 'PE_SELL'),
    (2026, 5, 12, 10, 15, 'CE_SELL'),
    (2026, 5, 14, 12, 15, 'PE_SELL'),
    (2026, 5, 29, 15, 15, 'CE_SELL'),
    (2026, 6, 12, 11, 15, 'PE_SELL')
]

# Sidebar Sidebar controls for Hedging and Buffer
st.sidebar.header("🔧 Strategy Settings")
hedging_on = st.sidebar.checkbox("Enable OTM Hedging (Spreads)", value=True, help="Margin aur Risk kam karne ke liye")
slippage_pct = st.sidebar.slider("Slippage Buffer per Trade (%)", 0.0, 0.2, 0.05, step=0.01)

if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Yahoo Finance se real data fetch ho raha hai..."):
        # Fetching data covering the schedule range
        nifty = yf.download("^NSEI", start="2025-07-01", end="2026-07-01", interval="1h")
        
    if nifty.empty:
        st.error("Data nahi mil paya. Internet connection check karein.")
    else:
        nifty.index = nifty.index.tz_localize(None)
        
        trade_log = []
        current_position = None
        total_pnl = 0
        
        for i in range(len(trade_schedule)):
            y, m, d, h, minute, trade_type = trade_schedule[i]
            target_time = datetime(y, m, d, h, minute)
            
            try:
                closest_idx = nifty.index.get_indexer([target_time], method='nearest')[0]
                spot_price = nifty.iloc[closest_idx]['Close']
                if isinstance(spot_price, pd.Series):
                    spot_price = float(spot_price.iloc[0])
                else:
                    spot_price = float(spot_price)
            except Exception:
                continue
                
            atm_strike = round(spot_price / 50) * 50
            
            # Agar purani position open hai, to pehle use SQUARE OFF/EXIT karein
            if current_position is not None:
                entry_type = current_position['type']
                entry_spot = current_position['spot']
                entry_strike = current_position['strike']
                entry_time = current_position['time']
                
                price_change = spot_price - entry_spot
                
                # Option delta calculation (~0.5 for ATM)
                if entry_type == 'CE_SELL':
                    raw_pnl = -price_change * 0.5
                else:
                    raw_pnl = price_change * 0.5
                
                # Theta Decay benefit model
                days_held = (target_time - entry_time).days
                theta_gain = entry_strike * 0.0015 * max(1, days_held) # 0.15% per day decay
                
                trade_pnl = raw_pnl + theta_gain
                
                # Applying Hedging adjustment if enabled
                if hedging_on:
                    trade_pnl *= 0.85  # Hedging premium safety reduction factor
                    
                # Slippage deduction
                trade_pnl -= (spot_price * (slippage_pct / 100))
                total_pnl += trade_pnl
                
                trade_log.append({
                    "Entry Date": entry_time.strftime('%Y-%m-%d %H:%M'),
                    "Exit Date": target_time.strftime('%Y-%m-%d %H:%M'),
                    "Trade Block": entry_type,
                    "ATM Strike": entry_strike,
                    "Nifty Entry Spot": round(entry_spot, 2),
                    "Nifty Exit Spot": round(spot_price, 2),
                    "Net PnL (Points)": round(trade_pnl, 2)
                })
                
            # Current loop state saved for the next turn
            current_position = {
                "time": target_time,
                "type": trade_type,
                "strike": atm_strike,
                "spot": spot_price
            }
            
        # UI Report Layout
        df_results = pd.DataFrame(trade_log)
        
        # Metrics Display
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Net PnL", f"{round(total_pnl, 2)} Points")
        win_rate = (df_results['Net PnL (Points)'] > 0).sum() / len(df_results) * 100
        col2.metric("📈 Win Rate", f"{round(win_rate, 2)}%")
        col3.metric("🔄 Total Loops Handled", len(df_results))
        
        st.markdown("---")
        st.subheader("📜 Detailed Trade Log (Step-by-Step Reversals)")
        
        # Function to color PnL rows
        def color_pnl(val):
            color = '#a2f3a2' if val > 0 else '#f3a2a2'
            return f'background-color: {color}'
            
        st.dataframe(df_results.style.map(color_pnl, subset=['Net PnL (Points)']), use_container_width=True)
