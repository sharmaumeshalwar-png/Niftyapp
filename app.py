import streamlit as st
import pandas as pd
import yfinance as yf

# Page Setup
st.set_page_config(layout="wide")
st.title("🚀 Nifty Discovery View")

# Sidebar - News side mein shift kar di
with st.sidebar:
    st.header("📢 Market Events")
    st.write("• 10:00 AM: RBI Policy")
    st.write("• 01:30 PM: Reliance Q1")

# Data fetch simple rakha hai
@st.cache_data(ttl=3600)
def get_simple_data():
    raw = yf.download("^NSEI", period="1mo", interval="1h")
    df = pd.DataFrame(index=raw.index)
    df['Price'] = raw['Close']
    return df

data = get_simple_data().sort_index(ascending=False)

# Table display
st.subheader("📋 Discovery Table (Drag columns to move)")
st.write("Date aur Price yahan dikhenge. Column headers ko pakad kar left-right shift karein.")

st.data_editor(
    data,
    use_container_width=True,
    height=600
)
