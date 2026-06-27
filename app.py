import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

st.title("Nifty Price Tracking (Kalman & Particle Filter)")

# 1. Nifty ka data download karo
st.write("Data download ho raha hai...")
data = yf.download('^NSEI', period='6mo', interval='1d')

# ---- SAFETY CHECK (ISSE ERROR NAHI AAYEGA) ----
if data.empty:
    st.error("Yahoo Finance se data nahi mil pa raha hai. Kripya thodi der baad try karein ya Ticker change karein.")
else:
    a = data['Close'].values.flatten()
    num_steps = len(a)

    # 2. Kalman Filter on 'a' (b)
    b = np.zeros(num_steps)
    x_est = a[0]  # Ab yeh safe hai kyunki data empty nahi hai
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

    # 5. Streamlit par Graph dikhana
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(a, label="a: Nifty Close", color='blue')
    ax1.plot(b, label="b: Kalman Filter", color='red', linestyle='--')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(c, label="c: Residual (a-b)", color='purple')
    ax2.plot(d, label="d: Particle Filter", color='green', linestyle='-.')
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig)
