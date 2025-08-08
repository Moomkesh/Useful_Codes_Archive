import numpy as np
import matplotlib.pyplot as plt

class LowPassFilter:
    def __init__(self):
        # Initialize dictionaries to store state for multiple instances
        self.output_signal_last = {}
        self.input_signal_last = {}
        self.a1 = {}
        self.a0 = {}
        self.b1 = {}
        self.b0 = {}

    def lp_filter(self, input_signal, dt, corner_freq, instance=0, i_status=0, reset=False, initial_value=None):
        """
        Discrete-time low-pass filter equivalent to the Fortran LPFilter function.
        
        Parameters:
        - input_signal (float): Current input signal value
        - dt (float): Time step [seconds]
        - corner_freq (float): Corner frequency [rad/s]
        - instance (int): Instance number for multiple filters
        - i_status (int): Simulation status (0: first call, 1: subsequent, -1: final)
        - reset (bool): Reset filter to initial value
        - initial_value (float, optional): Value to set when resetting
        
        Returns:
        - filtered output (float)
        """
        # Default initial value is the input signal if not provided
        initial_value = input_signal if initial_value is None else initial_value

        # Initialization or reset
        if i_status == 0 or reset:
            self.output_signal_last[instance] = initial_value
            self.input_signal_last[instance] = initial_value
            self.a1[instance] = 2 + corner_freq * dt
            self.a0[instance] = corner_freq * dt - 2
            self.b1[instance] = corner_freq * dt
            self.b0[instance] = corner_freq * dt

        # Compute filter output using the difference equation
        output = (1.0 / self.a1[instance]) * (
            -self.a0[instance] * self.output_signal_last[instance] +
            self.b1[instance] * input_signal +
            self.b0[instance] * self.input_signal_last[instance]
        )

        # Update states for next time step
        self.input_signal_last[instance] = input_signal
        self.output_signal_last[instance] = output

        return output

def plot_frequency_response(dt, corner_freq):
    # Frequency points for response
    freqs = np.logspace(-1, 2, 500)  # 0.1 to 100 Hz
    w = 2 * np.pi * freqs  # Angular frequency [rad/s]
    
    # Filter coefficients
    a1 = 2 + corner_freq * dt
    a0 = corner_freq * dt - 2
    b1 = corner_freq * dt
    b0 = corner_freq * dt
    
    # Compute frequency response H(e^jw)
    H = np.zeros(len(w), dtype=complex)
    for i, omega in enumerate(w):
        z = np.exp(1j * omega * dt)  # e^(jωT)
        H[i] = (b1 * z + b0) / (a1 * z + a0)
    
    # Plot magnitude response
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs, 20 * np.log10(np.abs(H)), label='Frequency Response')
    plt.axvline(corner_freq / (2 * np.pi), color='r', linestyle='--', label='Corner Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Low-Pass Filter Frequency Response')
    plt.legend()
    plt.grid(True)
    plt.show()

# Demonstration
def main():
    # Parameters
    dt = 0.01  # Time step [s]
    corner_freq = 200.0  # Corner frequency [rad/s]
    t = np.arange(0, 2, dt)  # Time vector [0, 2] seconds
    freq = 1.0  # Base signal frequency [Hz]
    
    # Generate a noisy signal: sine wave + random noise
    np.random.seed(42)  # For reproducibility
    signal = np.cos(2 * np.pi * freq * t) + 0.5 * np.random.normal(0, 1, len(t))
    
    # Initialize filter
    lpf = LowPassFilter()
    
    # Apply filter to the signal
    filtered_signal = np.zeros_like(signal)
    for i, x in enumerate(signal):
        i_status = 0 if i == 0 else 1  # First call: i_status = 0
        filtered_signal[i] = lpf.lp_filter(x, dt, corner_freq, instance=0, i_status=i_status)
    
    # Plot time-domain results
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label='Input (Noisy Signal)', alpha=0.7)
    plt.plot(t, filtered_signal, label='Filtered Output', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Low-Pass Filter: Input vs. Filtered Output')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot frequency response
    plot_frequency_response(dt, corner_freq)

if __name__ == "__main__":
    main()