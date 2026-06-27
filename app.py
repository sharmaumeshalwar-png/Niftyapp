import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

st.title("Nifty 1-Hour Price Tracking (Frozen Data)")

st.write("1 Jan 2025 se 2 Saal ka 1-Hour data download ho raha hai...")

# ---- DATA FREEZE CORRECTION HERE ----
# interval='1h' se 1-hour candle aayegi
# start aur end date se data 2 saal par freeze ho jayega
data = yf.download('^NSEI', start='2025-01-01', end='2027-01-01', interval='1h')

# ---- SAFETY CHECK ----
if data.empty:
    st.error("Yahoo Finance ne is date range ka 1-Hour data nahi diya. (Note: Yahoo intraday/1h data sirf pichle 2-3 mahino ka hi save rakhta hai). Aap test karne ke liye interval='1d' (Daily) try kar sakte hain.")
else:
    # Baki ka poora code same rahega
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
    num_particles = 300
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

    # 5. Streamlit Graph
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(a, label="a: Nifty Close (1h)", color='blue')
    ax1.plot(b, label="b: Kalman Filter", color='red', linestyle='--')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(c, label="c: Residual (a-b)", color='purple')
    ax2.plot(d, label="d: Particle Filter", color='green', linestyle='-.')
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig)
