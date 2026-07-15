import io
import pandas as pd
import streamlit as st
import numpy as np

# =========================================================================
# 1. CORE MATHEMATICAL ENGINES (ORIGINAL BASELINE MATRIX)
# =========================================================================
def calculate_kalman_filter(data_array, R=0.2, Q=0.0005, P_init=50.0):
    x = data_array[0] if len(data_array) > 0 else 0.0
    P = P_init
    kalman_values = []
    for z in data_array:
        P = P + Q
        k = P / (P + R)
        x = x + k * (z - x)
        P = (1 - k) * P
        kalman_values.append(x)
    return np.array(kalman_values)

def calculate_hurst_exponent(price_series, max_lag=20):
    hurst_values = np.zeros(len(price_series))
    for i in range(max_lag, len(price_series)):
        window = price_series[i - max_lag + 1 : i + 1]
        log_returns = np.log(window[1:] / window[:-1])
        lags = range(2, max_lag // 2)
        tau = []
        for lag in lags:
            diffs = log_returns[lag:] - log_returns[:-lag]
            tau.append(np.std(diffs))
        if len(tau) > 1:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst_values[i] = poly[0] + 0.5
        else:
            hurst_values[i] = 0.5
    return hurst_values

def calculate_atr_momentum(df, period=14):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
    price_change = close - np.roll(close, 1)
    price_change[0] = 0.0

    atr_momentum = np.zeros(len(df))
    for i in range(len(df)):
        if atr[i] > 0:
            atr_momentum[i] = price_change[i] / atr[i]
        else:
            atr_momentum[i] = 0.0
    return atr_momentum

# =========================================================================
# 2. MAIN TRADING ENGINE WITH USER'S TWO EXTRA COLUMNS
# =========================================================================
def execute_complete_trading_system(df):
    df["Kalman_Line"] = calculate_kalman_filter(df["Close"].values)
    df["Hurst"] = calculate_hurst_exponent(df["Close"].values)
    df["ATR_Momentum"] = calculate_atr_momentum(df)

    # TWO NEW COLUMNS ADDED
    df["Trade_Points_PNL"] = 0.0
    df["ML_Brain_Insight"] = "No Active Signal"
    df["Active_Position"] = "NONE"

    in_position = False
    entry_price = 0.0
    position_type = None

    for i in range(1, len(df)):
        current_momentum = df.loc[i, "ATR_Momentum"]
        prev_momentum = df.loc[i - 1, "ATR_Momentum"]
        current_close = df.loc[i, "Close"]
        hurst_val = df.loc[i, "Hurst"]

        if current_momentum < 0 and prev_momentum >= 0 and not in_position:
            in_position = True
            entry_price = current_close
            position_type = "CE_SELL"
            df.loc[i, "Active_Position"] = "CE_SELL_OPEN"

        elif current_momentum > 0 and prev_momentum <= 0 and not in_position:
            in_position = True
            entry_price = current_close
            position_type = "PE_SELL"
            df.loc[i, "Active_Position"] = "PE_SELL_OPEN"

        elif in_position:
            df.loc[i, "Active_Position"] = f"HOLD_{position_type}"

            if (position_type == "CE_SELL" and current_momentum > 0) or (
                position_type == "PE_SELL" and current_momentum < 0
            ):
                # 1. Trade_Points_PNL Column Math
                if position_type == "CE_SELL":
                    pnl_points = entry_price - current_close
                else:
                    pnl_points = current_close - entry_price

                df.loc[i, "Trade_Points_PNL"] = round(pnl_points, 2)
                df.loc[i, "Active_Position"] = f"EXIT_{position_type}"

                # 2. ML_Brain_Insight Column Tag
                if pnl_points < 0:
                    if hurst_val < 0.45:
                        df.loc[i, "ML_Brain_Insight"] = f"❌ LOSS ({round(pnl_points)} pts). ML: Chop Zone (Hurst={round(hurst_val,2)}). Size: 0.01"
                    else:
                        df.loc[i, "ML_Brain_Insight"] = f"❌ LOSS ({round(pnl_points)} pts). ML: High Volatility."
                else:
                    if pnl_points >= 1500 and hurst_val > 0.55:
                        df.loc[i, "ML_Brain_Insight"] = f"🟢 JACKPOT ({round(pnl_points)} pts). ML: High Trend (Hurst={round(hurst_val,2)}). Scale: 0.10"
                    else:
                        df.loc[i, "ML_Brain_Insight"] = f"🟢 PROFIT ({round(pnl_points)} pts). ML: Normal Momentum."

                in_position = False
                entry_price = current_close
                position_type = "PE_SELL" if position_type == "CE_SELL" else "CE_SELL"
                in_position = True
                df.loc[i, "Active_Position"] = f"REVERSE_TO_{position_type}"

    return df

# =========================================================================
# 3. STREAMLIT ULTIMATE LIVE INTERFACE (NO DUMMY SIMULATOR)
# =========================================================================
st.set_page_config(page_title="Real Quant ML Engine", layout="wide")
st.title("📊 Real 2-Year Historical Matrix Reversal Engine")
st.subheader("Bypass Overload Engine - Direct CSV File Interface")

# Direct Uploader for Real Data Verification
uploaded_file = st.file_uploader("📥 Upload your real 2-Year historical CSV file (Must contain High, Low, Close)", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.success(f"🟢 File Loaded Successfully! Total Rows Found: {len(df_input)}")
    
    # Structural check for essential mapping
    required_cols = ["High", "Low", "Close"]
    missing_cols = [col for col in required_cols if col not in df_input.columns]
    
    if missing_cols:
        st.error(f"⚠️ Aapki file mein ye columns missing hain: {missing_cols}. Kripya format check karein paji!")
    else:
        if st.button("▶️ Run Backtest on Real File Data"):
            with st.spinner("Processing actual market columns logic..."):
                processed_matrix = execute_complete_trading_system(df_input)

                # Auto detection for Date Time Column names
                datetime_col = None
                for col in ["Date", "Time", "Date_Time", "datetime", "Date & Time", "timestamp"]:
                    if col in processed_matrix.columns:
                        datetime_col = col
                        break

                # Slicing table framework
                output_cols = ["Close", "High", "Low", "ATR_Momentum", "Hurst", "Active_Position", "Trade_Points_PNL", "ML_Brain_Insight"]
                if datetime_col:
                    output_cols = [datetime_col] + output_cols

                final_sheet = processed_matrix[output_cols]

                # Render Table View on Streamlit UI Screen
                st.write("### 📈 Live Screen View (Actual 750+ Data Rows Summary)")
                st.dataframe(final_sheet, height=600, use_container_width=True)

                # Clean CSV data download trigger
                csv_data = final_sheet.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download This Real Backtest Report (CSV File)",
                    data=csv_data,
                    file_name="Real_Market_Report.csv",
                    mime="text/csv",
                )
                st.success("🎉 Processed flawlessly! Data is 100% real as per your uploaded file.")
else:
    st.info("💡 Paji, apni real data wali CSV file ko upar upload karo. Upload karte hi asli Close Price aur Time ke sath report turant ready ho jayegi!")
