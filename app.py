import numpy as np
import pandas as pd
import yfinance as yf

# =====================================================================
# 1. HEIKIN-ASHI CALCULATION
# =====================================================================
def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df_ha = df.copy()

    # HA Close = Average of OHLC
    df_ha["HA_Close"] = (
        df_ha["Open"] + df_ha["High"] + df_ha["Low"] + df_ha["Close"]
    ) / 4.0

    # HA Open = Average of Previous HA Open and Previous HA Close
    ha_open = np.zeros(len(df_ha))
    ha_open[0] = (df_ha["Open"].iloc[0] + df_ha["Close"].iloc[0]) / 2.0

    for i in range(1, len(df_ha)):
        ha_open[i] = (ha_open[i - 1] + df_ha["HA_Close"].iloc[i - 1]) / 2.0

    df_ha["HA_Open"] = ha_open
    df_ha["HA_High"] = df_ha[["High", "HA_Open", "HA_Close"]].max(axis=1)
    df_ha["HA_Low"] = df_ha[["Low", "HA_Open", "HA_Close"]].min(axis=1)

    return df_ha


# =====================================================================
# 2. CUSTOM KALMAN FILTER (Single & Dual State)
# =====================================================================
def apply_kalman_filter_custom(
    series: np.ndarray,
    initial_p: float = 50.0,
    q_val: float = 0.0005,
    r_val: float = 0.2,
) -> np.ndarray:
    """
    Applies custom 1D Kalman Filter to smooth time series data.
    """
    n = len(series)
    filtered = np.zeros(n)

    # Initial estimates
    x_hat = series[0] if n > 0 else 0.0
    p = initial_p

    for i in range(n):
        # Time Update (Predict)
        x_hat_minus = x_hat
        p_minus = p + q_val

        # Measurement Update (Correct)
        k = p_minus / (p_minus + r_val)
        x_hat = x_hat_minus + k * (series[i] - x_hat_minus)
        p = (1 - k) * p_minus

        filtered[i] = x_hat

    return filtered


# =====================================================================
# 3. LEAK-FREE ROLLING HURST EXPONENT
# =====================================================================
def calculate_rolling_hurst_leak_free(
    series: np.ndarray, window: int = 30
) -> pd.Series:
    """
    Calculates rolling Hurst exponent using R/S Analysis.
    Shifted by 1 bar to strictly prevent lookahead leak.
    """
    hurst_vals = np.full(len(series), 0.5)  # Default neutral 0.5

    for i in range(window, len(series)):
        sub_series = series[i - window : i]
        mean_val = np.mean(sub_series)
        deviations = sub_series - mean_val
        cum_deviations = np.cumsum(deviations)

        r = np.max(cum_deviations) - np.min(cum_deviations)
        s = np.std(sub_series, ddof=1)

        if s > 1e-8 and r > 1e-8:
            rs_ratio = r / s
            # Hurst approximation via log(R/S) / log(N)
            h = np.log(rs_ratio) / np.log(window)
            hurst_vals[i] = np.clip(h, 0.0, 1.0)
        else:
            hurst_vals[i] = 0.5

    # Shift by 1 to guarantee zero lookahead bias
    return pd.Series(hurst_vals).shift(1).fillna(0.5)


# =====================================================================
# 4. CORE ENGINE: HA-HAM FEATURES (Window 30 vs Window 200)
# =====================================================================
def compute_ha_ham_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Velocity, Acceleration, HA_HAM_30 and HA_HAM_200 using
    identical theoretical parameters on two distinct window horizons.
    """
    df_ha = compute_heikin_ashi(df_raw)
    ha_close = df_ha["HA_Close"].to_numpy().flatten()

    # Physics Velocity & Acceleration
    v_window = 5
    df_ha["Velocity_Speed"] = df_ha["Close"].diff(v_window) / v_window
    df_ha["Acceleration_GForce"] = (
        df_ha["Velocity_Speed"].diff(v_window) / v_window
    )

    # -------------------------------------------------------------
    # 1. HAM Window 30 (Fast Intraday Momentum)
    # -------------------------------------------------------------
    df_ha["Hurst_30"] = calculate_rolling_hurst_leak_free(ha_close, window=30)

    # Kalman Filter Stage 1 (Base Trend) & Stage 2 (Momentum Residual)
    kalman_30_base = apply_kalman_filter_custom(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_30_res = apply_kalman_filter_custom(
        ha_close - kalman_30_base, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    # Final HAM Value = Momentum Residual * Hurst Multiplier
    df_ha["HA_HAM_30"] = np.array(mom_30_res) * (
        df_ha["Hurst_30"].to_numpy() * 2.0
    )

    # -------------------------------------------------------------
    # 2. HAM Window 200 (Macro Structural Anchor - EXACT SAME THEORY)
    # -------------------------------------------------------------
    df_ha["Hurst_200"] = calculate_rolling_hurst_leak_free(
        ha_close, window=200
    )

    # Identical q_val and r_val parameters as Window 30
    kalman_200_base = apply_kalman_filter_custom(
        ha_close, initial_p=50.0, q_val=0.0005, r_val=0.2
    )
    mom_200_res = apply_kalman_filter_custom(
        ha_close - kalman_200_base, initial_p=0.50, q_val=0.001, r_val=0.1
    )
    # Final HAM Value = Momentum Residual * Hurst Multiplier
    df_ha["HA_HAM_200"] = np.array(mom_200_res) * (
        df_ha["Hurst_200"].to_numpy() * 2.0
    )

    return df_ha


# =====================================================================
# 5. EXECUTION & DISPLAY DRIVE
# =====================================================================
if __name__ == "__main__":
    print("Fetching Nifty 50 Data...")
    ticker = "^NSEI"

    # Fetching 60d of 5m data to ensure enough bars (>200) for clean warmup
    df_raw = yf.download(ticker, period="60d", interval="5m", progress=False)

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    print("Computing HA_HAM Engine (30 vs 200 Window)...")
    df_processed = compute_ha_ham_features(df_raw)

    # Displaying last 15 rows with focused columns
    cols_to_show = [
        "Close",
        "HA_Close",
        "Velocity_Speed",
        "HA_HAM_30",
        "HA_HAM_200",
    ]
    print("\n--- LATEST 15 BARS RESULT ---")
    print(df_processed[cols_to_show].tail(15).round(4))
