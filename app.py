import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Nifty Pure Notebook Analyzer", layout="wide")
st.title("📊 Nifty 50 Notebook Performance Engine (Strict Sequence Mode)")
st.write("🎯 **Aapki Custom Setting:** Strictly same row sequence as your hand-written notebook paper (Starting from 4th July 2025) + 25-Hour Pts Track.")

# =====================================================================
# EXACT NOTEBOOK SEQUENCE (Strictly ordered from top-to-bottom, left-to-right)
# =====================================================================
notebook_ordered_records = [
    # LEFT COLUMN (Starting exactly from 4th July 2025)
    {"Date": "2025-07-04", "Time": "12:15", "Type": "CE_SELL"},
    {"Date": "2025-07-04", "Time": "15:15", "Type": "CE_SELL"},
    {"Date": "2025-07-08", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-07-08", "Time": "13:15", "Type": "PE_SELL"},
    {"Date": "2025-07-08", "Time": "14:15", "Type": "CE_SELL"},
    {"Date": "2025-07-08", "Time": "15:15", "Type": "PE_SELL"},
    {"Date": "2025-07-09", "Time": "15:15", "Type": "CE_SELL"},
    {"Date": "2025-02-18", "Time": "10:15", "Type": "PE_SELL"}, # Note: Kept exactly where it is in your column block
    {"Date": "2025-08-26", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-09-04", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-09-04", "Time": "15:15", "Type": "CE_SELL"},
    {"Date": "2025-09-08", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-09-25", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-10-06", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-11-04", "Time": "14:15", "Type": "CE_SELL"},
    {"Date": "2025-11-11", "Time": "15:15", "Type": "PE_SELL"},
    {"Date": "2025-11-25", "Time": "15:15", "Type": "CE_SELL"},
    {"Date": "2025-11-26", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-12-02", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-12-05", "Time": "12:15", "Type": "PE_SELL"},
    {"Date": "2025-12-08", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-12-22", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-12-24", "Time": "09:15", "Type": "CE_SELL"},
    {"Date": "2026-01-01", "Time": "11:15", "Type": "PE_SELL"},
    {"Date": "2026-01-01", "Time": "12:15", "Type": "CE_SELL"},
    
    # RIGHT COLUMN (Starts from 1st Jan 2025 block)
    {"Date": "2025-01-01", "Time": "14:15", "Type": "PE_SELL"},
    {"Date": "2025-01-01", "Time": "15:15", "Type": "CE_SELL"},
    {"Date": "2025-01-02", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-01-06", "Time": "11:15", "Type": "CE_SELL"},
    {"Date": "2025-02-03", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-02-05", "Time": "14:15", "Type": "CE_SELL"},
    {"Date": "2025-02-05", "Time": "15:15", "Type": "PE_SELL"},
    {"Date": "2025-02-06", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-07-06", "Time": "12:15", "Type": "PE_SELL"},
    {"Date": "2025-02-06", "Time": "13:15", "Type": "CE_SELL"},
    {"Date": "2025-02-06", "Time": "14:15", "Type": "PE_SELL"},
    {"Date": "2025-02-13", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-02-16", "Time": "15:15", "Type": "PE_SELL"},
    {"Date": "2025-02-19", "Time": "12:15", "Type": "CE_SELL"},
    {"Date": "2025-02-23", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-02-23", "Time": "13:15", "Type": "CE_SELL"},
    {"Date": "2025-02-23", "Time": "15:15", "Type": "PE_SELL"},
    {"Date": "2025-02-24", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-04-08", "Time": "10:15", "Type": "PE_SELL"},
    {"Date": "2025-05-12", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-05-14", "Time": "12:15", "Type": "PE_SELL"},
    {"Date": "2025-05-29", "Time": "15:15", "Type": "CE_SELL"},
    {"Date": "2025-06-12", "Time": "11:15", "Type": "PE_SELL"}
]

def get_nearest_thursday(date_str):
    base_date = pd.to_datetime(date_str)
    days_ahead = 3 - base_date.weekday()
    if days_ahead < 0: 
        days_ahead += 7
    return (base_date + pd.Timedelta(days=days_ahead)).strftime('%Y-%m-%d')

with st.spinner("Syncing Nifty Historical Timeline..."):
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    
    # Flatten MultiIndex immediately
    df = pd.DataFrame(index=raw_df.index)
    for col_name in ['Open', 'High', 'Low', 'Close']:
        if col_name in raw_df.columns:
            column_data = raw_df[col_name]
            df[col_name] = column_data.iloc[:, 0].astype(float) if isinstance(column_data, pd.DataFrame) else column_data.astype(float)

    df.index = pd.to_datetime(df.index)
    
    # Native Volatility Processing Engine
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Native_Volatility'] = df['Log_Ret'].rolling(window=25).std() * np.sqrt(24 * 365) * 100
    df['Native_Volatility'] = df['Native_Volatility'].ffill().fillna(14.0)

    df['Str_Date'] = df.index.strftime('%Y-%m-%d')
    df['Str_Hour'] = df.index.strftime('%H')

processed_sequential_logs = []
serial_number = 1

for trade_row in notebook_ordered_records:
    t_date = trade_row['Date']
    t_hour = trade_row['Time'].split(':')[0]
    t_type = trade_row['Type']
    
    entry_slice = df[(df['Str_Date'] == t_date) & (df['Str_Hour'] == t_hour)]
    
    if len(entry_slice) > 0:
        entry_idx = entry_slice.index[0]
        entry_price = float(entry_slice['Close'].iloc[0])
        current_vix = float(entry_slice['Native_Volatility'].iloc[0])
        
        all_subsequent_data = df.loc[entry_idx:]
        
        if len(all_subsequent_data) >= 25:
            exit_slice = all_subsequent_data.iloc[24] 
            exit_price = float(exit_slice['Close'])
            exit_time_str = exit_slice.name.strftime('%Y-%m-%d %H:%M')
        else:
            exit_slice = all_subsequent_data.iloc[-1]
            exit_price = float(exit_slice['Close'])
            exit_time_str = exit_slice.name.strftime('%Y-%m-%d %H:%M') + " (Open)"

        # Directional Profit/Loss Vector Calculation
        if t_type == "PE_SELL": 
            points_delta = exit_price - entry_price
            action_label = "🟢 PE SELL"
        else: 
            points_delta = entry_price - exit_price
            action_label = "🔴 CE SELL"

        calculated_expiry = get_nearest_thursday(t_date)
        days_to_expiry = (pd.to_datetime(calculated_expiry) - pd.to_datetime(t_date)).days
        
        if current_vix > 17.0:
            recommended_strike = "Far OTM Contract"
            vix_regime = "💥 HIGH"
        elif current_vix < 13.0:
            recommended_strike = "ATM / Close OTM Contract"
            vix_regime = "😴 LOW"
        else:
            recommended_strike = "Standard OTM Contract"
            vix_regime = "⚖️ BALANCED"

        processed_sequential_logs.append({
            "Sr No.": serial_number,
            "Notebook Entry Time": f"{t_date} {trade_row['Time']}",
            "Strategy Type": action_label,
            "Entry Nifty Spot": round(entry_price, 2),
            "Exit Time (25h)": exit_time_str,
            "Exit Nifty Spot": round(exit_price, 2),
            "Points Captured": round(points_delta, 2),
            "Nearest Weekly Expiry": calculated_expiry,
            "Days Left to Exp": days_to_expiry,
            "VIX Regime": vix_regime,
            "Optimized Option Selection": recommended_strike
        })
        serial_number += 1

if len(processed_sequential_logs) > 0:
    # CRITICAL CHANGE: No default dynamic date sorting. Keeps your exact list alignment!
    analysis_master_df = pd.DataFrame(processed_sequential_logs)

    total_points_captured = analysis_master_df['Points Captured'].sum()
    win_ratio = (analysis_master_df['Points Captured'] > 0).sum() / len(analysis_master_df) * 100

    # Summary Metrics Block
    st.subheader("📊 Notebook Strategy Metrics Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Points Captured", f"{round(total_points_captured, 2)} Pts")
    col2.metric("Historical Strategy Win Ratio", f"{round(win_ratio, 2)}%")
    col3.metric("Evaluated Sheet Records Count", f"{len(analysis_master_df)} Trades")

    st.markdown("---")
    st.subheader("📋 Performance Audit Grid View (Strict Sequence Mode)")
    st.dataframe(analysis_master_df, use_container_width=True, height=750)
else:
    st.warning("Data boundary alignment tracking empty.")
