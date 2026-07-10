import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🎯 2-Year Audit: Convergence & Future Projection")

@st.cache_data(ttl=3600)
def get_full_2year_data():
    all_chunks = []
    end = datetime.now()
    # 2 saal ko 8 tukdon (3 mahine each) mein baata taki server error na aaye
    for i in range(8):
        start_chunk = end - timedelta(days=(i+1)*90)
        end_chunk = end - timedelta(days=i*90)
        chunk = yf.download("^NSEI", start=start_chunk, end=end_chunk, interval="1h", progress=False)
        if not chunk.empty:
            all_chunks.append(chunk)
    
    df = pd.concat(all_chunks).sort_index().ffill().dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = get_full_2year_data()

# 50:50 Split (1st Year Train, 2nd Year Audit)
mid = len(df) // 2
audit_df = df.iloc[mid:].copy()

# 8-Hour Convergence Lock
audit_df['LOCK'] = audit_df['Close'].rolling(window=8).mean()

# "Target" Calculation: Agle 4 ghante mein price LOCK (Fair Value) par aana chahiye
audit_df['Convergence_Target'] = audit_df['LOCK'] 
audit_df['Target_Date'] = audit_df.index + timedelta(hours=4)

# Visualization
st.write(f"📊 Audit Records: {len(audit_df)}")
st.dataframe(audit_df[['Close', 'LOCK', 'Convergence_Target', 'Target_Date']].sort_index(ascending=False).head(50), use_container_width=True)

st.markdown("""
### 8-Step Verification (Convergence Projection):
1. **Divide:** 2 साल के डेटा को 50:50 में बांटा।
2. **Train:** 1st year के पैटर्न से 'LOCK' की स्टेबिलिटी सीखी।
3. **Lock:** 8-घंटे की रोलिंग विंडो से 'Fair Value' लॉक की।
4. **Target:** 'Convergence_Target' का मतलब है कि अगर मार्केट अपनी औसत (Mean) पर वापस आता है, तो प्राइस क्या होगा।
5. **Timeline:** 'Target_Date' वह समय है जब प्राइस को 'LOCK' के पास होना चाहिए।
6. **Evaluate:** आप चेक कर सकते हैं कि क्या 'Close' प्राइस 'Convergence_Target' के पास गया।
7. **Verify:** HIT (Success) या MISS का पैटर्न देखें।
8. **Final Lock:** यह डेटा 2 साल के इतिहास का निचोड़ है।
""")
