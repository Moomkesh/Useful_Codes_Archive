import numpy as np
import matplotlib.pyplot as plt

def lp_filter(input_signal, dt, corner_freq, prev_input, prev_output):
    """Simple first-order low-pass filter for a single signal."""
    # Filter coefficients using bilinear transform
    a1 = 2 + corner_freq * dt
    a0 = corner_freq * dt - 2
    b1 = corner_freq * dt
    b0 = corner_freq * dt
    
    # Compute filtered output
    output = (-a0 * prev_output + b1 * input_signal + b0 * prev_input) / a1
    
    return output, input_signal, output

# Parameters
dt = 0.01  # Time step (100 Hz sampling)
corner_freq = 0.17952  # Corner frequency in rad/s (0.0286 Hz)
t = np.arange(0, 100, dt)  # Time vector (0 to 100 seconds for 0.01 Hz signal)
signal = np.sin(2 * np.pi * 0.01 * t) + 0.5 * np.random.randn(len(t))  # 0.01 Hz signal + noise

# Initialize
prev_input = 0
prev_output = 0
filtered = []

# Apply filter
for x in signal:
    output, _, _ = lp_filter(x, dt, corner_freq, prev_input, prev_output)
    filtered.append(output)
    prev_input = x
    prev_output = output

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Original Signal (0.01 Hz + Noise)', alpha=0.7)
plt.plot(t, filtered, label='Filtered Signal (Low-Pass, 0.0286 Hz)', linewidth=2)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Low-Pass Filter for Yaw Controller: Original vs Filtered Signal')
plt.legend()
plt.grid(True)
plt.show()