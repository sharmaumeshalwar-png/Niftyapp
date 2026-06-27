import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 1. Nifty ka real data download karo (Yeh ban gaya input 'a')
print("Nifty 50 ka data download ho raha hai, kripya ruken...")
data = yf.download('^NSEI', period='6mo', interval='1d')
a = data['Close'].values.flatten()
num_steps = len(a)

# 2. Kalman Filter on 'a' (Yeh ban gaya input 'b')
b = np.zeros(num_steps)
x_est = a[0]  # Pehla price starting point hai
P = 1.0       # Prediction error
Q = 0.1       # Process noise
R = 2.0       # Measurement noise

for t in range(num_steps):
    x_pred = x_est
    P_pred = P + Q
    K = P_pred / (P_pred + R)  # Kalman Gain
    x_est = x_pred + K * (a[t] - x_pred)
    P = (1 - K) * P_pred
    b[t] = x_est

# 3. Calculate 'c' (c = a - b)
c = a - b

# 4. Particle Filter on 'c' (Yeh ban gaya input 'd')
num_particles = 300
particles = np.random.normal(c[0], 1, num_particles)
weights = np.ones(num_particles) / num_particles
d = np.zeros(num_steps)

for t in range(num_steps):
    # Predict step
    particles = particles + np.random.normal(0, 0.5, num_particles)
    
    # Update weights (Kitna paas hai actual 'c' ke)
    distance = particles - c[t]
    weights = np.exp(- (distance ** 2) / (2 * 1.5 ** 2)) + 1e-300
    weights /= np.sum(weights)  # Normalize
    
    # State Estimate 'd'
    d[t] = np.sum(particles * weights)
    
    # Simple Resampling
    if 1.0 / np.sum(weights ** 2) < num_particles / 2:
        idx = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
        particles = particles[idx]
        weights = np.ones(num_particles) / num_particles

# 5. Dono graphs ko screen par dikhana
print("Data processed! Graph generate ho raha hai...")
plt.figure(figsize=(12, 6))

# Pehla Graph: Nifty aur Kalman
plt.subplot(2, 1, 1)
plt.plot(a, label="a: Nifty Close Price", color='blue')
plt.plot(b, label="b: Kalman Filter", color='red', linestyle='--')
plt.title("Part 1: Nifty & Kalman")
plt.legend()
plt.grid(True)

# Dusra Graph: Residual aur Particle Method
plt.subplot(2, 1, 2)
plt.plot(c, label="c: Residual (a - b)", color='purple')
plt.plot(d, label="d: Particle Filter on c", color='green', linestyle='-.')
plt.title("Part 2: Residual & Particle Method")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
