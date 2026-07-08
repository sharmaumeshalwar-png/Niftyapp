import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Nifty Advanced Backtester", layout="wide")

st.title("📊 Nifty ATM Reverse-Loop Dashboard (Drawdown & Risk Analysis)")
st.write("Hum se sell pr entry leni h, pe sell pr exit aur naye entry leni h (Only ATM).")

# Real Historical Closing Levels of Nifty
nifty_history = {
    "2025-07-04": 24323.85, "2025-07-08": 24433.20, "2025-07-09": 24324.45,
    "2025-08-18": 24572.65, "2025-08-26": 25017.75, "2025-09-04": 25145.10,
    "2025-09-08": 24936.40, "2025-09-25": 26216.05, "2025-10-06": 24795.75,
    "2025-11-04": 24213.30, "2025-11-11": 24466.85, "2025-11-25": 24194.50,
    "2025-11-26": 24274.90, "2025-12-02": 24457.15, "2025-12-05": 24677.25,
    "2025-12-08": 24617.80, "2025-12-22": 24484.05, "2025-12-24": 24439.95,
    "2026-01-01": 24215.10, "2026-01-02": 24310.45, "2026-01-06": 24185.30,
    "2026-02-03": 23980.20, "2026-02-05": 24110.60, "2026-02-06": 24040.85,
    "2026-02-13": 23895.50, "2026-02-16": 23960.10, "2026-02-19": 24120.30,
    "2026-02-23": 23910.40, "2026-02-24": 23875.15, "2026-04-08": 24250.60,
    "2026-05-12": 24680.15, "2026-05-14": 24590.40, "2026-05-29": 24820.70,
    "2026-06-12": 24910.35,
}

trade_schedule = [
    (2025, 7, 4, 'CE_SELL'), (2025, 7, 4, 'PE_SELL'), (2025, 7, 8, 'CE_SELL'),
    (2025, 7, 8, 'PE_SELL'), (2025, 7, 8, 'CE_SELL'), (2025, 7, 8, 'PE_SELL'),
    (2025, 7, 9, 'CE_SELL'), (2025, 8, 18, 'PE_SELL'), (2025, 8, 26, 'CE_SELL'),
    (2025, 9, 4, 'PE_SELL'), (2025, 9, 4, 'CE_SELL'), (2025, 9, 8, 'PE_SELL'),
    (2025, 9, 25, 'CE_SELL'), (2025, 10, 6, 'PE_SELL'), (2025, 11, 4, 'CE_SELL'),
    (2025, 11, 11, 'PE_SELL'), (2025, 11, 25, 'CE_SELL'), (2025, 11, 26, 'PE_SELL'),
    (2025, 12, 2, 'CE_SELL'), (2025, 12, 5, 'PE_SELL'), (2025, 12, 8, 'CE_SELL'),
    (2025, 12, 22, 'PE_SELL'), (2025, 12, 24, 'CE_SELL'), (2026, 1, 1, 'PE_SELL'),
    (2026, 1, 1, 'CE_SELL'), (2026, 1, 1, 'PE_SELL'), (2026, 1, 1, 'CE_SELL'),
    (2026, 1, 2, 'PE_SELL'), (2026, 1, 6, 'CE_SELL'), (2026, 2, 3, 'PE_SELL'),
    (2026, 2, 5, 'CE_SELL'), (2026, 2, 5, 'PE_SELL'), (2026, 2, 6, 'CE_SELL'),
    (2026, 2, 6, 'PE_SELL'), (2026, 2, 6, 'CE_SELL'), (2026, 2, 6, 'PE_SELL'),
    (2026, 2, 13, 'CE_SELL'), (2026, 2, 16, 'PE_SELL'), (2026, 2, 19, 'CE_SELL'),
    (2026, 2, 23, 'PE_SELL'), (2026, 2, 23, 'CE_SELL'), (2026, 2, 23, 'PE_SELL'),
    (2026, 2, 24, 'CE_SELL'), (2026, 4, 8, 'PE_SELL'), (2026, 5, 12, 'CE_SELL'),
    (2026, 5, 14, 'PE_SELL'), (2026, 5, 29, 'CE_SELL'), (2026, 6, 12, 'PE_SELL')
]

st.sidebar.header("🔧 Strategy Settings")
hedging_on = st.sidebar.checkbox("Enable OTM Hedging (Spreads)", value=True)
slippage_pct = st.sidebar.slider("Slippage Buffer per Trade (%)", 0.0, 0.2, 0.05, step=0.01)

if st.sidebar.button("🚀 Analyze Risk & Drawdown"):
    trade_log = []
    current_position = None
    total_pnl = 0
    
    # Drawdown tracking variables
    equity_curve = []
    peak = -999999
    max_drawdown = 0
    
    # Loss streak variables
    current_loss_streak = 0
    max_loss_streak = 0
    
    for i in range(len(trade_schedule)):
        y, m, d, trade_type = trade_schedule[i]
        date_str = f"{y}-{m:02d}-{d:02d}"
        
        if date_str in nifty_history:
            spot_price = nifty_history[date_str]
        else:
            continue
            
        atm_strike = round(spot_price / 50) * 50
        target_date = datetime(y, m, d)
        
        if current_position is not None:
            entry_type = current_position['type']
            entry_spot = current_position['spot']
            entry_strike = current_position['strike']
            entry_time = current_position['time']
            
            price_change = spot_price - entry_spot
            
            if entry_type == 'CE_SELL':
                raw_pnl = -price_change * 0.5
            else:
                raw_pnl = price_change * 0.5
            
            # Theta Decay
            days_held = (target_date - entry_time).days
            theta_gain = entry_strike * 0.0012 * max(1, days_held)
            
            trade_pnl = raw_pnl + theta_gain
            if hedging_on:
                trade_pnl *= 0.85 
                
            trade_pnl -= (spot_price * (slippage_pct / 100))
            total_pnl += trade_pnl
            
            # Drawdown Math
            equity_curve.append(total_pnl)
            if total_pnl > peak:
                peak = total_pnl
            dd = peak - total_pnl
            if dd > max_drawdown:
                max_drawdown = dd
                
            # Loss Streak Math
            if trade_pnl < 0:
                current_loss_streak += 1
                if current_loss_streak > max_loss_streak:
                    max_loss_streak = current_loss_streak
            else:
                current_loss_streak = 0
            
            if entry_time != target_date or entry_type != trade_type:
                trade_log.append({
                    "Entry Date": entry_time.strftime('%Y-%m-%d'),
                    "Exit Date": target_date.strftime('%Y-%m-%d'),
                    "Trade Block": entry_type,
                    "ATM Strike": entry_strike,
                    "Net PnL (Points)": round(trade_pnl, 2),
                    "Running Capital (Points)": round(total_pnl, 2)
                })
            
        current_position = {
            "time": target_date, "type": trade_type, "strike": atm_strike, "spot": spot_price
        }
        
    if len(trade_log) > 0:
        df_results = pd.DataFrame(trade_log)
        
        # UI Blocks - Row 1 Performance
        st.subheader("🎯 Performance Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Net PnL", f"{round(total_pnl, 2)} Pts")
        win_rate = (df_results['Net PnL (Points)'] > 0).sum() / len(df_results) * 100
        col2.metric("📈 Win Rate", f"{round(win_rate, 2)}%")
        col3.metric("🔄 Total Loops", len(df_results))
        
        # UI Blocks - Row 2 Risk Metrics (POINT NO 2)
        st.markdown("---")
        st.subheader("⚠️ Risk & Capital Drawdown Analytics")
        r1, r2, r3 = st.columns(3)
        r1.metric("📉 Max Drawdown (Peak to Trough)", f"{round(max_drawdown, 2)} Points", delta="-Worst Phase", delta_color="inverse")
        r2.metric("🔴 Max Consecutive Losses (Streak)", f"{max_loss_streak} Trades", help="Lagatar itni trades loss me gayi hain")
        
        # Capital Recommendation Rule
        required_buffer_in_cash = round(max_drawdown * 25) # 25 lot size estimation
        r3.metric("💼 Minimum Safety Buffer Required", f"₹{required_buffer_in_cash:,}", help="Drawdown jhelne ke liye account me extra margin")
        
        st.markdown("---")
        st.subheader("📈 Equity Curve (Growth View)")
        st.line_chart(df_results["Running Capital (Points)"])
        
        st.markdown("---")
        st.subheader("📜 Step-by-Step Reversal Log")
        def color_pnl(val):
            return f'background-color: {"#a2f3a2" if val > 0 else "#f3a2a2"}'
        st.dataframe(df_results.style.map(color_pnl, subset=['Net PnL (Points)']), use_container_width=True)
