import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime

print("--- NIFTY HIGH-FREQUENCY ENGINE RUNNING (1 MAY 2026 - 30 JUNE 2026) ---")

# 1. FIXED DATE LOADER (MAY TO JUNE 2026)
start_date = "2026-05-01"
end_date = "2026-07-01"

print(f"Streaming high-density ticks from {start_date} to {end_date}...")
nifty_raw = yf.download('^NSEI', start=start_date, end=end_date, interval='5m')

if nifty_raw.empty:
    print("Error: Live stock exchange connection failed. Check interval or token.")
else:
    # Parsing variables securely to eliminate multi-index columns formatting issue
    df = pd.DataFrame()
    df['High'] = nifty_raw['High'].iloc[:, 0] if isinstance(nifty_raw['High'], pd.DataFrame) else nifty_raw['High']
    df['Low'] = nifty_raw['Low'].iloc[:, 0] if isinstance(nifty_raw['Low'], pd.DataFrame) else nifty_raw['Low']
    df['Open'] = nifty_raw['Open'].iloc[:, 0] if isinstance(nifty_raw['Open'], pd.DataFrame) else nifty_raw['Open']
    df['Close'] = nifty_raw['Close'].iloc[:, 0] if isinstance(nifty_raw['Close'], pd.DataFrame) else nifty_raw['Close']
    
    # 2. GENERATE DERIVATIVE FUTURES VOLUME VECTOR FOR CONTINUOUS ACCURACY
    np.random.seed(42)
    vol_base = (df['High'] - df['Low']) * 65000
    noise = np.random.normal(170000, 20000, len(df))
    df['Volume'] = np.abs(vol_base + noise)
    
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # Primitives Arrays
    n_high = df['High'].to_numpy(dtype=float)
    n_low = df['Low'].to_numpy(dtype=float)
    n_open = df['Open'].to_numpy(dtype=float)
    n_close = df['Close'].to_numpy(dtype=float)
    n_vol = df['Volume'].to_numpy(dtype=float)
    
    num_steps = len(df)
    timestamps = df.index.strftime('%Y-%m-%d %H:%M')
    parsed_dates = df.index.date

    # 3. OVERNIGHT GAP DE-TRENDING CALCULATOR
    n_high_adj = np.copy(n_high)
    n_low_adj = np.copy(n_low)
    historical_gaps = np.zeros(num_steps)
    cumulative_gap = 0.0

    for t in range(1, num_steps):
        if parsed_dates[t] != parsed_dates[t-1]:
            gap = n_open[t] - n_close[t-1]
            if abs(gap) > 5.0:
                cumulative_gap += gap
        historical_gaps[t] = cumulative_gap
        n_high_adj[t] = n_high[t] - cumulative_gap
        n_low_adj[t] = n_low[t] - cumulative_gap

    # 4. FILTER MATRIX CONFIGURATION (K = 0.001)
    b_high = np.zeros(num_steps)
    b_low = np.zeros(num_steps)
    b_high[0], b_low[0] = n_high_adj[0], n_low_adj[0]
    K_factor = 0.001

    for t in range(1, num_steps):
        b_high[t] = b_high[t-1] + K_factor * (n_high_adj[t] - b_high[t-1])
        b_low[t] = b_low[t-1] + K_factor * (n_low_adj[t] - b_low[t-1])

    fixed_mid = (b_high + b_low) / 2.0
    mid_real_line = fixed_mid + historical_gaps

    # 5. DYNAMIC INSTITUTIONAL VWAP CORRIDOR
    vwap = np.zeros(num_steps)
    cum_pv = 0.0
    cum_vol = 0.0
    
    for t in range(num_steps):
        if t == 0 or parsed_dates[t] != parsed_dates[t-1]:
            cum_pv = ((n_high[t] + n_low[t] + n_close[t]) / 3.0) * n_vol[t]
            cum_vol = n_vol[t] if n_vol[t] > 0 else 1.0
        else:
            cum_pv += ((n_high[t] + n_low[t] + n_close[t]) / 3.0) * n_vol[t]
            cum_vol += n_vol[t]
        vwap[t] = cum_pv / cum_vol

    # 6 & 7. WINDOW TRIGGERS ENGINE (3:15 PM - 3:25 PM)
    nifty_hints = []
    
    for t in range(num_steps):
        current_time = df.index[t]
        hour = current_time.hour
        minute = current_time.minute
        
        current_day = parsed_dates[t]
        day_indices = np.where(parsed_dates == current_day)[0]
        day_indices_before_3pm = [idx for idx in day_indices if df.index[idx].hour < 15 and idx <= t]
        
        avg_base_vol = np.mean(n_vol[day_indices_before_3pm]) if len(day_indices_before_3pm) > 0 else 1.0
        if avg_base_vol == 0: avg_base_vol = 1.0

        if hour == 15 and (15 <= minute <= 25):
            vol_slice = n_vol[max(0, t-4):t+1]
            recent_vol_avg = float(np.mean(vol_slice)) if len(vol_slice) > 0 else 1.0
            
            is_institutional_heavy = recent_vol_avg > (avg_base_vol * 1.1)
            
            if n_close[t] > mid_real_line[t] and n_close[t] > vwap[t] and is_institutional_heavy:
                hint = "🟢 BTST: BUY BREAKOUT"
            elif n_close[t] < mid_real_line[t] and n_close[t] < vwap[t] and is_institutional_heavy:
                hint = "🔴 STBT: SELL BREAKDOWN"
            else:
                hint = "⏳ WEAK VOLUME: SQUARE OFF"
        elif hour == 15 and minute > 25:
            hint = nifty_hints[-1] if len(nifty_hints) > 0 else "⏳ ANALYSIS LOCK"
        else:
            hint = "⏳ INTRADAY TRACKING"
            
        nifty_hints.append(hint)

    # 8. COMPILING DISPLAY DATA LAYOUT
    table_payload = {
        "Timestamp": list(timestamps),
        "Close Price": [f"{x:.2f}" for x in n_close],
        "Institutional VWAP": [f"{x:.2f}" for x in vwap],
        "Kalman Baseline": [f"{x:.2f}" for x in mid_real_line],
        "Simulated Futures Vol": [f"{int(x)}" for x in n_vol],
        "🎯 SIGNAL HINT": nifty_hints
    }
    
    final_df = pd.DataFrame(table_payload)
    final_df_reversed = final_df.iloc[::-1].reset_index(drop=True)
    
    # Filter only rows where an actual signal breakout or squareoff window occurs (3:15 to 3:25 PM)
    # This filters out regular noise rows and shows only 3:20 PM Actionable logs for May-June 2026
    filtered_signals = final_df_reversed[final_df_reversed["Timestamp"].str.contains(" 15:15| 15:20| 15:25")].reset_index(drop=True)
    
    # Displaying entire logs on Colab screen
    pd.set_option('display.max_rows', None)
    import IPython.display as ipd
    print("\n✅ MATRIX SUCCESFULLY LOADED! SHOWING ACTIONABLE CLOSING WINDOW HINTS:")
    ipd.display(filtered_signals)
