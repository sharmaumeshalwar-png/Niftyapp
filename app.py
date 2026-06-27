import numpy as np
import matplotlib.pyplot as plt

# Step 1: Seed for reproducibility
np.random.seed(42)
num_steps = 200

# Step 2: Generate Dummy Nifty Close Price (a)
a = 22000 + np.cumsum(np.random.normal(0, 15, num_steps))

# Step 3: Kalman Filter on 'a' to get 'b'
b = np.zeros(num_steps)
x_est = a[0]  # Initial state estimate
P = 1.0       # Initial estimation error covariance
Q = 0.5       # Process noise covariance
R = 5.0       # Measurement noise covariance

for t in range(num_steps):
    # Prediction Update
    x_pred = x_est
    P_pred = P + Q
    
    # Measurement Update
    K = P_pred / (P_pred + R)  # Kalman Gain
    x_est = x_pred + K * (a[t] - x_pred)
    P = (1 - K) * P_pred
    b[t] = x_est

# Step 4: Calculate c = a - b
c = a - b

# Step 5 & 6: Particle Filter on 'c' to get 'd'
num_particles = 500
particles = np.random.normal(c[0], 2, num_particles)
weights = np.ones(num_particles) / num_particles

d = np.zeros(num_steps)
process_noise = 0.5
measurement_noise = 1.5

for t in range(num_steps):
    # Predict step for particles
    particles = particles + np.random.normal(0, process_noise, num_particles)
    
    # Update weights based on distance to actual 'c' value (Likelihood)
    distance = particles - c[t]
    weights = np.exp(- (distance ** 2) / (2 * measurement_noise ** 2))
    weights += 1e-300  # Avoid division by zero
    weights /= np.sum(weights)  # Normalize
    
    # Estimate State (d) - Weighted average of particles
    d[t] = np.sum(particles * weights)
    
    # Step 7: Resampling (Systematic Resampling)
    eff_particles = 1.0 / np.sum(weights ** 2)
    if eff_particles < num_particles / 2:
        cumulative_sum = np.cumsum(weights)
        positions = (np.random.random() + np.arange(num_particles)) / num_particles
        idx = np.zeros(num_particles, dtype=int)
        i, j = 0, 0
        while i < num_particles:
            if positions[i] < cumulative_sum[j]:
                idx[i] = j
                i += 1
            else:
                j += 1
        particles = particles[idx]
        weights = np.ones(num_particles) / num_particles

# Step 8: Plotting the Results
plt.figure(figsize=(14, 8))

# Subplot 1: Nifty Price & Kalman Filter
plt.subplot(2, 1, 1)
plt.plot(a, label="a: Nifty Close Price", color='blue', alpha=0.6)
plt.plot(b, label="b: Kalman Filter on 'a'", color='red', linestyle='--')
plt.title("Nifty Price Tracking using Kalman Filter")
plt.legend()
plt.grid(True)

# Subplot 2: Residual 'c' & Particle Filter 'd'
plt.subplot(2, 1, 2)
plt.plot(c, label="c: Residual (a - b)", color='purple', alpha=0.6)
plt.plot(d, label="d: Particle Filter on 'c'", color='green', linestyle='-.')
plt.title("Residual Tracking using Particle Filter")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
