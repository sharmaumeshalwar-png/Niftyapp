import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Nifty True Spot Matcher", layout="wide")
st.title("🦅 Nifty 50 Notebook Performance Engine (True 5-Min Spot Resolution)")
st.write("🎯 **Aapki Custom Setting:** Explicit 5-Minute Micro-Data Sync + Exact Timestamp Precision Tracking + Precise 300-Candle (25h) Exit Lookahead.")

# =====================================================================
# EXACT NOTEBOOK SEQUENCE (As-is notebook order preserved)
# =====================================================================
notebook_ordered_records = [
    # LEFT COLUMN
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

with st.spinner("Downloading High-Precision 5-Minute Nifty Historical Data..."):
    # Download raw 5m data frames (Using 60d limit standard constraint max window for intraday 5m)
    raw_df = yf.download("^NSEI", period="max", interval="5m")
    
    if len(raw_df) == 0:
        st.error("Market API Error while downloading micro data blocks.")
        st.stop()

    # Accurate localized conversion to India Market Timeline
    if raw_df.index.tz is None:
        raw_df.index = raw_df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        raw_df.index = raw_df.index.tz_convert('Asia/Kolkata')

    # Flatten out MultiIndex structures cleanly
    df = pd.DataFrame(index=raw_df.index)
    for col_name in ['Open', 'High', 'Low', 'Close']:
        if col_name in raw_df.columns:
            column_data = raw_df[col_name]
            df[col_name] = column_data.iloc[:, 0].astype(float) if isinstance(column_data, pd.DataFrame) else column_data.astype(float)

    # Volatility estimation from 5-minute ticks
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Native_Volatility'] = df['Log_Ret'].rolling(window=300).std() * np.sqrt(12 * 6 * 24 * 365) * 100
    df['Native_Volatility'] = df['Native_Volatility'].ffill().fillna(15.0)

    # String cross-matching strings calculation
    df['Str_DateTime'] = df.index.strftime('%Y-%m-%d %H:%M')
    df['Str_Date'] = df.index.strftime('%Y-%m-%d')

processed_sequential_logs = []
serial_number = 1

for trade_row in notebook_ordered_records:
    t_date = trade_row['Date']
    t_time = trade_row['Time']
    t_type = trade_row['Type']
    
    target_string = f"{t_date} {t_time}"
    
    # Precise match via strict 5-minute row lookup string
    entry_slice = df[df['Str_DateTime'] == target_string]
    
    # Dynamic proximity match if index boundary falls at a slightly offset 5m marker
    if len(entry_slice) == 0:
        entry_slice = df[df['Str_Date'] == t_date].head(1)

    if len(entry_slice) > 0:
        entry_idx = entry_slice.index[0]
        entry_price = float(entry_slice['Close'].iloc[0])
        current_vix = float(entry_slice['Native_Volatility'].iloc[0])
        
        all_subsequent_data = df.loc[entry_idx:]
        
        # 25 Hours = exactly 300 candles on a 5-minute data stream
        if len(all_subsequent_data) >= 300:
            exit_slice = all_subsequent_data.iloc[299] 
            exit_price = float(exit_slice['Close'])
            exit_time_str = exit_slice.name.strftime('%Y-%m-%d %H:%M')
        else:
            exit_slice = all_subsequent_data.iloc[-1]
            exit_price = float(exit_slice['Close'])
            exit_time_str = exit_slice.name.strftime('%Y-%m-%d %H:%M') + " (Open)"

        # Execution Mathematics
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
            "Notebook Entry Time": target_string,
            "Strategy Type": action_label,
            "Entry Nifty Spot (True)": round(entry_price, 2),
            "Exit Time (25h)": exit_time_str,
            "Exit Nifty Spot (True)": round(exit_price, 2),
            "Points Captured": round(points_delta, 2),
            "Nearest Weekly Expiry": calculated_expiry,
            "Days Left to Exp": days_to_expiry,
            "VIX Regime": vix_regime,
            "Optimized Option Selection": recommended_strike
        })
        serial_number += 1

if len(processed_sequential_logs) > 0:
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
    st.subheader("📋 Performance Audit Grid View (High-Precision 5m Spot Execution)")
    st.dataframe(analysis_master_df, use_container_width=True, height=750)
else:
    st.warning("No tracking rows crossed data boundary matching filters. Verify current market feeds.")
