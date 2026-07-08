import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Nifty Sheet Performance & Expiry Analyzer", layout="wide")
st.title("🦅 Nifty 50 Institutional Sheet Performance Engine & Expiry Optimizer")
st.write("🎯 **Core Objective:** Analytical evaluation of your hand-written sheet trades + 25-Hour execution point tracking + Adaptive Expiry Selection Matrix with Native Volatility correlation.")

# =====================================================================
# HARDCODED IMAGE DATA MATRICES (Strictly Mapped)
# =====================================================================
sheet_records = [
    # Left Column (2025)
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
    # Left Column (2026 Entries)
    {"Date": "2026-01-01", "Time": "11:15", "Type": "PE_SELL"},
    {"Date": "2026-01-01", "Time": "12:15", "Type": "CE_SELL"},
    # Right Column Mappings (2025 Data blocks)
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

hints_df = pd.DataFrame(sheet_records)

# =====================================================================
# STRATEGY UTILITY FUNCTIONS
# =====================================================================
def get_nearest_thursday(date_str):
    """Calculates weekly Thursday contract target anchors for execution processing."""
    base_date = pd.to_datetime(date_str)
    days_ahead = 3 - base_date.weekday()
    if days_ahead < 0: 
        days_ahead += 7
    return (base_date + pd.Timedelta(days=days_ahead)).strftime('%Y-%m-%d')

with st.spinner("Extracting Market Time Series and Volatility Arrays..."):
    # Ingest 2 full years data for baseline tracking stabilization
    raw_df = yf.download("^NSEI", period="2y", interval="1h")
    if len(raw_df) == 0:
        st.error("Market API Endpoint Error. Please reload the computational application.")
        st.stop()

    df = pd.DataFrame(index=raw_df.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = raw_df[col].iloc[:, 0] if isinstance(raw_df[col], pd.DataFrame) else raw_df[col]

    df.index = pd.to_datetime(df.index)
    
    # Internal Volatility Calculation Engine (Alternative VIX proxy)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Native_Volatility'] = df['Log_Ret'].rolling(window=25).std() * np.sqrt(24 * 365) * 100
    df['Native_Volatility'] = df['Native_Volatility'].ffill().fillna(15.0)

    # Cross-matching reference index columns
    df['Str_Date'] = df.index.strftime('%Y-%m-%d')
    df['Str_Hour'] = df.index.strftime('%H')

# =====================================================================
# CALCULATION LOOP & VECTOR CORRELATION ENGINE
# =====================================================================
processed_logs = []

for _, trade_row in hints_df.iterrows():
    t_date = trade_row['Date']
    t_hour = trade_row['Time'].split(':')[0]
    t_type = trade_row['Type']
    
    # Match structural price layers using the soft-hour matcher block
    entry_slice = df[(df['Str_Date'] == t_date) & (df['Str_Hour'] == t_hour)]
    
    if len(entry_slice) > 0:
        entry_idx = entry_slice.index[0]
        entry_price = float(entry_slice['Close'].values[0])
        current_vix = float(entry_slice['Native_Volatility'].values[0])
        
        # Pull positional lookahead location for exit check (25 operational hours out)
        all_subsequent_data = df.loc[entry_idx:]
        
        if len(all_subsequent_data) >= 25:
            exit_slice = all_subsequent_data.iloc[24] # Lookahead operational coordinate index 25
            exit_price = float(exit_slice['Close'].values[0])
            exit_time_str = exit_slice.index.strftime('%Y-%m-%d %H:%M')[0]
        else:
            exit_slice = all_subsequent_data.iloc[-1]
            exit_price = float(exit_slice['Close'].values[0])
            exit_time_str = exit_slice.index.strftime('%Y-%m-%d %H:%M')[0] + " (Live/Open)"

        # Directional Point Math Execution Vector
        if t_type == "PE_SELL": # Bullish Trade Strategy
            points_delta = exit_price - entry_price
            action_label = "🟢 PE SELL [Bullish]"
        else: # CE_SELL Bearish Trade Strategy
            points_delta = entry_price - exit_price
            action_label = "🔴 CE SELL [Bearish]"

        # Expiry Strategic Selection Rules
        calculated_expiry = get_nearest_thursday(t_date)
        days_to_expiry = (pd.to_datetime(calculated_expiry) - pd.to_datetime(t_date)).days
        
        # Adaptive Premium Selection Filters based on Volatility Matrix
        if current_vix > 18.0:
            recommended_strike = "Far OTM (Buffer for high swings)"
            vix_regime = "💥 HIGH (Sell Premium Expansion Zone)"
        elif current_vix < 13.0:
            recommended_strike = "ATM / Close OTM (Premium compression zone)"
            vix_regime = "😴 LOW (Tight Spreads Recommended)"
        else:
            recommended_strike = "Standard OTM (1-1.5% out of boundary)"
            vix_regime = "⚖️ BALANCED"

        processed_logs.append({
            "Entry Time": f"{t_date} {trade_row['Time']}",
            "Type": action_label,
            "Entry Nifty": round(entry_price, 2),
            "Exit Time (25h)": exit_time_str,
            "Exit Nifty": round(exit_price, 2),
            "Points Earned": round(points_delta, 2),
            "Estimated Expiry": calculated_expiry,
            "Days to Expiry": days_to_expiry,
            "Native Volatility": round(current_vix, 2),
            "VIX Regime": vix_regime,
            "Optimized Strike Rule": recommended_strike
        })

# Format tracking metrics back into display DataFrame
analysis_master_df = pd.DataFrame(processed_logs)
analysis_master_df = analysis_master_df.sort_values(by="Entry Time", ascending=False)

# Total Performance Aggregators
total_points_captured = analysis_master_df['Points Earned'].sum()
win_ratio = (analysis_master_df['Points Earned'] > 0).sum() / len(analysis_master_df) * 100

# =====================================================================
# DASHBOARD INTERFACE COMPILING
# =====================================================================
st.subheader("📊 Sheet Strategy Metrics Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Gross Points Captured", f"{round(total_points_captured, 2)} Pts")
col2.metric("Historical Strategy Win Ratio", f"{round(win_ratio, 2)}%")
col3.metric("Evaluated Sheet Records Count", f"{len(analysis_master_df)} Trades")

st.markdown("---")
st.subheader("💡 Strategic Rules Checklist for Executing Sheet Hints Successfully")
st.info("""
1. **Expiry Contract Parameter:** Target date se check karein; agar **Days to Expiry <= 2** bache hain, toh **Next Weekly Contract** me trade shift karein taaki theta decay smooth mile aur gamma trap se bach sakein.
2. **Volatility Correction Filter:** Agar **Native Volatility High (💥)** chal rahi ho, toh strike boundary ko current spots se minimum **1.5% out-of-the-money (OTM)** push karein. Low volatility ranges me **ATM ya close OTM** select karein.
3. **Execution Window Validation:** Sheet entry logs hit hote hi price delta ko underlying asset directional momentum ke against mat lagayein. Check structural trends using the layout metrics table beneath.
""")

st.subheader("📋 Performance Audit Grid View")
st.dataframe(analysis_master_df, use_container_width=True, height=650)
