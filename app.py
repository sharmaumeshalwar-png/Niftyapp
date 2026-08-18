# =====================================================================
# DISPLAY MATRIX & METRICS (PREDICT SLICE ONLY - LAST 100 CANDLES)
# =====================================================================
df_predict_out = df.iloc[split_idx:].copy()

# Table display ke liye sirf last 100 candles select karein
display_df_full = df_predict_out.iloc[::-1]
display_df_100 = display_df_full.head(100).copy()
display_df_100.index = display_df_100.index.strftime("%Y-%m-%d %H:%M IST")

latest_candle = display_df_100.iloc[0]
latest_time = display_df_100.index[0]

st.info(
    f"📊 **Data Partition Summary:** Total = {total_candles:,} Candles |"
    f" **Learn (Trained)** = {split_idx:,} | **Predict (Out-of-Sample)** ="
    f" {len(df_predict_out):,}"
)

st.markdown(
    f"### 🔒 **LAST LOCKED CANDLE (50% PREDICT WINDOW):** `{latest_time}`"
)

# Metrics Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Nifty Close Price", f"₹{latest_candle['Close']:,.2f}")
col2.metric("Base HAM Normal", f"{latest_candle['HAM_Normal']:.2f}")
col3.metric("🌌 Scale Factor (a)", f"{latest_candle['HAM_Expansion_a']:.4f}")
col4.metric(
    f"🔭 Hubble Vel (σ={gaussian_sigma})",
    f"{latest_candle['HAM_Hubble_Vel_v']:.2f} km/s",
)
col5.metric(
    "🚀 Cosmic Accel (ä)", f"{latest_candle['HAM_Cosmic_Accel_a_dotdot']:.4e}"
)

st.divider()

# Interactive Data Frame
st.subheader(
    f"📋 Recent 100 Candles Matrix (Out-of-Sample Predict Window)"
)

# Checkbox for full historical view
show_all = st.checkbox("Show all 2,500+ Predict Candles")
final_display = display_df_full.copy() if show_all else display_df_100

st.dataframe(
    final_display,
    column_config={
        "Close": st.column_config.NumberColumn(
            "Nifty Close (₹)", format="₹%.2f"
        ),
        "HA_Close": st.column_config.NumberColumn(
            "HA Close (₹)", format="₹%.2f"
        ),
        "Hurst_Normal": st.column_config.NumberColumn(
            "Hurst", format="%.2f"
        ),
        "HAM_Normal": st.column_config.NumberColumn(
            "Base HAM Normal (Kalman)", format="%.2f"
        ),
        "HAM_Expansion_a": st.column_config.NumberColumn(
            "🌌 Scale Factor a(t)", format="%.4f"
        ),
        "HAM_Hubble_Vel_v": st.column_config.NumberColumn(
            f"🔭 Hubble Vel (Gaussian σ={gaussian_sigma})", format="%.2f"
        ),
        "HAM_Cosmic_Accel_a_dotdot": st.column_config.NumberColumn(
            "🚀 Cosmic Accel (ä)", format="%.4e"
        ),
        "HAM_HeikinAshi": st.column_config.NumberColumn(
            "HAM HA Signal (Kalman)", format="%.2f"
        ),
        "HAM_Hint": st.column_config.TextColumn("💡 HAM Hint Dynamic"),
        "HAM_Diff_Kalman": st.column_config.NumberColumn(
            "📊 HAM Diff (Kalman)", format="%.2f"
        ),
        "HAM_Velocity": st.column_config.NumberColumn(
            "⚡ Velocity (Kalman Q=1e-6)", format="%.4f"
        ),
        "HAM_Acceleration": st.column_config.NumberColumn(
            "🚀 Acceleration (Δ2)", format="%.4f"
        ),
        "Flip_Status": st.column_config.TextColumn(
            "🎯 50:50 State Lock (Predict)"
        ),
    },
    use_container_width=True,
    height=600,
)
