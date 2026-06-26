import pandas as pd
import streamlit as st
from tvdatafeed import Interval, TvDatafeed

st.write("🔄 Fetching Live Nifty 1-Hour Data via TradingView Feed...")


@st.cache_data(ttl=900)  # Data ko 15 mins tak cache karega
def fetch_nifty_live():
    try:
        # Anonymous login (Bina user id password ke)
        tv = TvDatafeed()
        # Nifty 50 index data from NSE
        df_tv = tv.get_hist(
            symbol="NIFTY",
            exchange="NSE",
            interval=Interval.in_1_hour,
            n_bars=1000,
        )
        if df_tv is not None and not df_tv.empty:
            # Columns rename karna taaki aapka baki ka Kalman code smoothly chal sake
            df_tv = df_tv.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                }
            )
            return df_tv
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


df = fetch_nifty_live()

if df.empty:
    st.error("❌ Data source temporaily down. Real data nahi mil pa rha.")
    st.stop()
else:
    st.success(f"✅ Live Data Connected! Rows: {len(df)}")
    # Aapka adaptive kalman aur 'c' loop iske niche perfect chalega!
