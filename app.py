import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd

st.title("Nifty 1-Hour Price Data Table (Frozen)")

st.write("1 Jan 2025 se 2 Saal ka 1-Hour data table mein load ho raha hai...")

# 1. Data Download (Interval 1-Hour, Frozen Dates)
data = yf.download('^NSEI', start='2025-01-01', end='2027-01-01', interval='1h')

# ---- SAFETY CHECK ----
if data.empty:
    st.error("Yahoo Finance ne is date range ka 1-Hour data nahi diya. Aap test karne ke liye interval='1d' (Daily) karke check kar sakte hain.")
else:
    # Timestamps (Dates) nikalne ke liye
    timestamps = data.index.strftime('%Y-%m-%d %H:%M')
    
    a = data['Close'].values.flatten()
    num_steps = len(a)

    # 2. Kalman Filter on 'a' (b)
    b = np.zeros(num_steps)
    x_est = a[0]
    P = 1.0       
    Q = 0.1       
    R = 2.0       

    for t in range(num_steps):
        x_pred = x_est
        P_pred = P + Q
        K = P_pred / (P_pred + R)  
        x_est = x_pred + K * (a[t] - x_pred)
        P = (1 - K) * P_pred
        b[t] = x_est

    # 3. Calculate 'c' (c = a - b)
    c = a - b

    # 4. Particle Filter on 'c' (d)
    num_particles = 100  # Table ke liye speed badhane ko particles kam kiye hain
    particles = np.random.normal(c[0], 1, num_particles)
    weights = np.ones(num_particles) / num_particles
    d = np.zeros(num_steps)

    for t in range(num_steps):
        particles = particles + np.random.normal(0, 0.5, num_particles)
        distance = particles - c[t]
        weights = np.exp(- (distance ** 2) / (2 * 1.5 ** 2)) + 1e-300
        weights /= np.sum(weights)  
        d[t] = np.sum(particles * weights)
        
        if 1.0 / np.sum(weights ** 2) < num_particles / 2:
            idx = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
            particles = particles[idx]
            weights = np.ones(num_particles) / num_particles

    # ---- STEP 5: CREATING THE SIMPLE TABLE ----
    df_table = pd.DataFrame({
        'Date & Time': timestamps,
        'a: Nifty Close': np.round(a, 2),
        'b: Kalman Filter': np.round(b, 2),
        'c: Residual (a - b)': np.round(c, 2),
        'd: Particle Filter': np.round(d, 2)
    })

    # Step 6: Set Date as Index
    df_table.set_index('Date & Time', inplace=True)

    # Step 7: Streamlit Table View
    st.subheader("Final Processed Data (a, b, c, d)")
    st.dataframe(df_table, use_container_width=True)
    
    # Step 8: Success Info
    st.success("Table ke upar right side mein download button hai, wahan se aap isko Excel/CSV mein nikal sakte hain!")
