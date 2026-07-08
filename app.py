import yfinance as tf
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Aapka bataya hua complete date aur time schedule (Sample Segment for execution)
# Format: (Year, Month, Day, Hour, Minute, Trade_Type)
trade_schedule = [
    (2025, 7, 4, 12, 15, 'CE_SELL'),
    (2025, 7, 4, 15, 15, 'PE_SELL'),
    (2025, 7, 8, 10, 15, 'CE_SELL'),
    (2025, 7, 8, 13, 15, 'PE_SELL'),
    (2025, 7, 8, 14, 15, 'CE_SELL'),
    (2025, 7, 8, 15, 15, 'PE_SELL'),
    (2025, 7, 9, 15, 15, 'CE_SELL'),
    # Aap isme upar di gayi list ke baaki saare dates isi format me add kar sakte hain
]

def backtest_system():
    print("Fetching Nifty Historical 1-Hour Data from Yahoo Finance...")
    # Nifty 50 data download (1h interval max 2 saal tak ka milta hai)
    nifty = tf.download("^NSEI", start="2025-07-01", end="2026-07-01", interval="1h")
    
    if nifty.empty:
        print("Data nahi mila! Please check internet connection.")
        return

    nifty.index = nifty.index.tz_localize(None) # Timezone remove karne ke liye
    
    trade_log = []
    current_position = None
    total_pnl = 0
    
    print("\n--- Processing Trades According to Your Schedule ---")
    
    for i in range(len(trade_schedule)):
        y, m, d, h, minute, trade_type = trade_schedule[i]
        target_time = datetime(y, m, d, h, minute)
        
        # Yahoo Finance ke nearest 1-hr candle search karna
        try:
            closest_idx = nifty.index.get_indexer([target_time], method='nearest')[0]
            spot_price = nifty.iloc[closest_idx]['Close']
            # Multi-level index bypass karne ke liye float conversion
            if isinstance(spot_price, pd.Series):
                spot_price = float(spot_price.iloc[0])
            else:
                spot_price = float(spot_price)
        except Exception:
            continue
            
        atm_strike = round(spot_price / 50) * 50
        
        # Agar pehle se koi position open hai, toh use pehle EXIT (Square off) karenge
        if current_position is not None:
            entry_type = current_position['type']
            entry_spot = current_position['spot']
            entry_strike = current_position['strike']
            
            # Intraday / Positional Movement calculation
            price_change = spot_price - entry_spot
            
            # Premium Estimation Model (Delta ~0.5 for ATM with Hedging Consideration)
            # Hedging ke sath intrinsic value aur time decay simulate kiya hai
            if entry_type == 'CE_SELL':
                raw_pnl = -price_change * 0.5  # Market upar gaya toh CE Sell me loss
            else:
                raw_pnl = price_change * 0.5   # Market upar gaya toh PE Sell me profit
                
            # Time Decay (Theta Benefit adding approx 0.1% per day)
            days_held = (target_time - current_position['time']).days
            theta_gain = entry_strike * 0.001 * max(1, days_held)
            trade_pnl = raw_pnl + theta_gain
            
            # Slippage aur brokerage safety buffer deduction
            trade_pnl -= (spot_price * 0.0005) 
            total_pnl += trade_pnl
            
            trade_log.append({
                "Exit_Time": target_time,
                "Type": entry_type,
                "Strike": entry_strike,
                "PnL_Points": round(trade_pnl, 2)
            })
            
        # Nayi Entry lena
        current_position = {
            "time": target_time,
            "type": trade_type,
            "strike": atm_strike,
            "spot": spot_price
        }
        
    # Final Report Presentation
    df_results = pd.DataFrame(trade_log)
    print("\n================ BACKTEST REPORT ================")
    print(df_results.to_string())
    print("=================================================")
    print(f"Total Accumulated PnL: {round(total_pnl, 2)} Nifty Points")
    print(f"Win Rate: {round((df_results['PnL_Points'] > 0).sum() / len(df_results) * 100, 2)}%")
    print("=================================================")

# Script ko run karne ke liye
backtest_system()
