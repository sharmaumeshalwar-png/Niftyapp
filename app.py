import os
import numpy as np
import pandas as pd


# =========================================================================
# 1. CORE MATHEMATICAL ENGINES
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
# 2. MAIN PRODUCTION TRADING & ML INSIGHT CORE
# =========================================================================
def execute_complete_trading_system(df):
    df["Kalman_Line"] = calculate_kalman_filter(df["Close"].values)
    df["Hurst"] = calculate_hurst_exponent(df["Close"].values)
    df["ATR_Momentum"] = calculate_atr_momentum(df)

    # NAYE DONO COLUMNS INJECT KARTE HAIN
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

        # CE Sell Trigger
        if current_momentum < 0 and prev_momentum >= 0 and not in_position:
            in_position = True
            entry_price = current_close
            position_type = "CE_SELL"
            df.loc[i, "Active_Position"] = "CE_SELL_OPEN"

        # PE Sell Trigger
        elif current_momentum > 0 and prev_momentum <= 0 and not in_position:
            in_position = True
            entry_price = current_close
            position_type = "PE_SELL"
            df.loc[i, "Active_Position"] = "PE_SELL_OPEN"

        # Running State Loop (The Flip Engine)
        elif in_position:
            df.loc[i, "Active_Position"] = f"HOLD_{position_type}"

            if (position_type == "CE_SELL" and current_momentum > 0) or (
                position_type == "PE_SELL" and current_momentum < 0
            ):
                # 1. NEW COLUMN: Points PNL calculation
                if position_type == "CE_SELL":
                    pnl_points = entry_price - current_close
                else:
                    pnl_points = current_close - entry_price

                df.loc[i, "Trade_Points_PNL"] = round(pnl_points, 2)
                df.loc[i, "Active_Position"] = f"EXIT_{position_type}"

                # 2. NEW COLUMN: ML Brain Learning Logger
                if pnl_points < 0:
                    if hurst_val < 0.45:
                        df.loc[i, "ML_Brain_Insight"] = (
                            f"❌ LOSS ({round(pnl_points)} pts). ML Learned: False Flip due to Chop Zone (Hurst={round(hurst_val,2)}). Action: Next time reduce size to 0.01 BTC."
                        )
                    else:
                        df.loc[i, "ML_Brain_Insight"] = (
                            f"❌ LOSS ({round(pnl_points)} pts). ML Learned: High Volatility Anomaly. Action: Move exit to 15-min micro baseline."
                        )
                else:
                    if pnl_points >= 1500 and hurst_val > 0.55:
                        df.loc[i, "ML_Brain_Insight"] = (
                            f"🟢 JACKPOT ({round(pnl_points)} pts). ML Learned: High Trend Persistence Confirmed (Hurst={round(hurst_val,2)}). Action: Aggressive Trailing & scale up to 0.10 BTC."
                        )
                    else:
                        df.loc[i, "ML_Brain_Insight"] = (
                            f"🟢 PROFIT ({round(pnl_points)} pts). ML Learned: Normal Momentum Reversal. Action: Maintain standard position size."
                        )

                # Auto-Reverse on Flip Instantly
                in_position = False
                entry_price = current_close
                position_type = "PE_SELL" if position_type == "CE_SELL" else "CE_SELL"
                in_position = True
                df.loc[i, "Active_Position"] = f"REVERSE_TO_{position_type}"

    return df


# =========================================================================
# 3. SMART DATA-STREAM INTERFACE (AUTO-SIMULATOR RUNNER)
# =========================================================================
if __name__ == "__main__":
    file_name = "historical_2year_data.csv"

    # Agar file hai toh use karega, nahi toh 750 rows generate karega
    if os.path.exists(file_name):
        print(f"🟢 Real 2-Year Database File '{file_name}' found! Loading rows...")
        df_input = pd.read_csv(file_name)
    else:
        print(
            f"⚠️ Note: '{file_name}' nahi mili. Activating Auto-Simulator Engine..."
        )
        print("🔄 Generating perfect 750 Rows historical structure for testing...\n")
        np.random.seed(42)
        base_price = 64000
        price_movement = np.random.normal(5, 300, 750)
        simulated_closes = np.cumsum(price_movement) + base_price

        df_input = pd.DataFrame(
            {
                "High": simulated_closes + np.random.uniform(50, 200, 750),
                "Low": simulated_closes - np.random.uniform(50, 200, 750),
                "Close": simulated_closes,
            }
        )

    # Process the entire 750 grid
    processed_matrix = execute_complete_trading_system(df_input)

    # Full terminal view activation configuration
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 850)
    pd.set_option("display.width", 1200)

    # Pure print execution targeting full 750 corridor rows
    print(
        processed_matrix[
            [
                "Close",
                "ATR_Momentum",
                "Hurst",
                "Active_Position",
                "Trade_Points_PNL",
                "ML_Brain_Insight",
            ]
        ].to_string()
    )

    # Excel/CSV generation audit trail
    processed_matrix.to_csv("ML_Quant_Backtest_Output.csv", index=False)
    print(
        "\n💾 Data saved successfully! Check 'ML_Quant_Backtest_Output.csv' file in your folder paji!"
    )
