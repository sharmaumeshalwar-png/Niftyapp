import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Nifty Robust Spot Engine", layout="wide")
st.title("🦅 Nifty 50 Notebook Performance Engine (Robust Daily Mode)")
st.write("🎯 **Aapki Custom Setting:** Multi-Year Unrestricted Daily Data Sync + Exact Notebook Sequence Mapping + 4-Day (25h) Exit Lookahead Grid.")

# =====================================================================
# EXACT NOTEBOOK SEQUENCE (Preserved as-is from your sheet columns)
# =====================================================================
notebook_ordered_records = [
    # LEFT COLUMN (Starting from 4th July 2025)
    {"Date": "2025-07-04", "Time": "12:15", "Type": "CE_SELL"},
    {"Date": "2025-07-04", "Time": "15:15", "Type": "CE_SELL"},
    {"Date": "2025-07-08", "Time": "10:15", "Type": "CE_SELL"},
    {"Date": "2025-07-08", "Time": "13:15", "Type": "PE_SELL"},
    {"Date": "2025-07-08", "Time": "14:15", "Type": "CE_SELL"},
    {"Date": "2025-07-08", "Time": "15:15", "Type": "PE_SELL"},
    {"Date": "2025-07-09", "Time": "15:15", "Type": "CE_SELL"},
    {"Date": "2025-02-18", "Time": "10:15", "Type": "PE_SELL"},
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
    
    # RIGHT COLUMN
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

with st.spinner("Downloading Stable Multi-Year Daily Series (No Data Drops)..."):
    # Download raw daily historical data (unrestricted timeline window)
    raw_df = yf.download("^NSEI", start="2024-11-01", end="2026-03-01", interval="1d")
    
    if len(raw_df) == 0:
        st.error("Historical pricing feed link broken. Refresh dashboard layout.")
        st.stop()

    # Complete Flattening of multi-level columns immediately
    df = pd.DataFrame(index=raw_df.index)
    for col_name in ['Open', 'High', 'Low', 'Close']:
        if col_name in raw_df.columns:
            column_series = raw_df[col_name]
            df[col_name] = column_series.iloc[:, 0].astype(float) if isinstance(column_series, pd.DataFrame) else column_series.astype(float)

    df.index = pd.to_datetime(df.index)
    
    # Calculate daily systemic historical volatility
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Daily_Volatility'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
    df['Daily_Volatility'] = df['Daily_Volatility'].ffill().fillna(14.0)

    df['Str_Date'] = df.index.strftime('%Y-%m-%d')

processed_logs = []
serial_num = 1

for trade_row in notebook_ordered_records:
    t_date = trade_row['Date']
    t_time = trade_row['Time']
    t_type = trade_row['Type']
    
    # Precise day alignment check
    entry_slice = df[df['Str_Date'] == t_date]
    
    # Soft match approach if sheet date falls on a weekend/holiday
    if len(entry_slice) == 0:
        target_dt = pd.to_datetime(t_date)
        # Scan next proximate trading window
        for offset in range(1, 5):
            alternate_str = (target_dt + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
            entry_slice = df[df['Str_Date'] == alternate_str]
            if len(entry_slice) > 0:
                break

    if len(entry_slice) > 0:
        entry_idx = entry_slice.index[0]
        entry_price = float(entry_slice['Close'].iloc[0])
        current_vix = float(entry_slice['Daily_Volatility'].iloc[0])
        
        all_subsequent_data = df.loc[entry_idx:]
        
        # 25 Hours equivalent = exactly 4 trading days out
        if len(all_subsequent_data) >= 5:
            exit_slice = all_subsequent_data.iloc[4]
            exit_price = float(exit_slice['Close'])
            exit_time_str = exit_slice.name.strftime('%Y-%m-%d')
        else:
            exit_slice = all_subsequent_data.iloc[-1]
            exit_price = float(exit_slice['Close'])
            exit_time_str = exit_slice.name.strftime('%Y-%m-%d') + " (Open Block)"

        # Performance Extraction Vector calculations
        if t_type == "PE_SELL":
            points_delta = exit_price - entry_price
            action_label = "🟢 PE SELL [Bullish]"
        else:
            points_delta = entry_price - exit_price
            action_label = "🔴 CE SELL [Bearish]"

        calculated_expiry = get_nearest_thursday(t_date)
        days_to_expiry = (pd.to_datetime(calculated_expiry) - pd.to_datetime(t_date)).days
        
        if current_vix > 16.5:
            recommended_strike = "Far OTM (1.5% Boundary)"
            vix_regime = "💥 HIGH"
        elif current_vix < 12.5:
            recommended_strike = "ATM / Close OTM Spreads"
            vix_regime = "😴 LOW"
        else:
            recommended_strike = "Standard OTM Contract"
            vix_regime = "⚖️ BALANCED"

        processed_logs.append({
            "Sr No.": serial_num,
            "Notebook Strategy Time": f"{t_date} {t_time}",
            "Strategy Type": action_label,
            "Nifty Entry Spot (Close)": round(entry_price, 2),
            "Exit Time (4 Days)": exit_time_str,
            "Nifty Exit Spot (Close)": round(exit_price, 2),
            "Points Captured": round(points_delta, 2),
            "Nearest Weekly Expiry": calculated_expiry,
            "Days to Expiry": days_to_expiry,
            "Market Volatility": round(current_vix, 2),
            "VIX Regime": vix_regime,
            "Optimized Option Strike": recommended_strike
        })
        serial_num += 1

if len(processed_logs) > 0:
    analysis_master_df = pd.DataFrame(processed_logs)

    total_points_captured = analysis_master_df['Points Captured'].sum()
    win_ratio = (analysis_master_df['Points Captured'] > 0).sum() / len(analysis_master_df) * 100

    # Summary Metrics Interface Dashboard
    st.subheader("📊 Notebook Strategy Metrics Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Points Captured", f"{round(total_points_captured, 2)} Pts")
    col2.metric("Historical Strategy Win Ratio", f"{round(win_ratio, 2)}%")
    col3.metric("Evaluated Sheet Records Count", f"{len(analysis_master_df)} Trades")

    st.markdown("---")
    st.subheader("📋 Performance Audit Grid View (Unrestricted Stable Daily Mode)")
    st.dataframe(analysis_master_df, use_container_width=True, height=750)
else:
    st.warning("Data matching pipeline returned empty matrix array.")
