import numpy as np
import matplotlib.pyplot as plt

def create_step_waveform(t, steps, step_times, shift=0):
    """
    Create a step waveform and its shifted version, then plot both.
    
    Parameters:
    t : array-like, time points where waveform is evaluated
    steps : array-like, step values
    step_times : array-like, times when steps occur
    shift : float, time shift for the shifted waveform (default=0)
    
    Returns:
    wd : numpy array, original waveform values at time points t
    wd_shifted : numpy array, shifted waveform values at time points t
    """
    # Create original waveform
    wd = np.zeros_like(t)
    for i in range(1, len(step_times)):
        mask = (t >= step_times[i-1]) & (t < step_times[i])
        wd[mask] = steps[i-1]
    wd[t >= step_times[-1]] = steps[-1]
    
    # Create shifted waveform
    wd_shifted = np.zeros_like(t)
    shifted_times = [t + shift for t in step_times]
    for i in range(1, len(shifted_times)):
        mask = (t >= shifted_times[i-1]) & (t < shifted_times[i])
        wd_shifted[mask] = steps[i-1]
    wd_shifted[t >= shifted_times[-1]] = steps[-1]
    
    # Plot both waveforms
    plt.figure(figsize=(10, 6))
    plt.step(t, wd, label='Original Waveform', where='post')
    plt.step(t, wd_shifted, label=f'Shifted Waveform (by {shift})', where='post')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Step Waveform and Shifted Waveform')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return wd, wd_shifted

# Example usage
if __name__ == "__main__":
    t = np.linspace(0, 700, 1000)
    steps = [0, 10, 20, 30, 10]
    step_times = [0, 100, 200, 400, 500, 600]
    shift = -50
    waveform, waveform_shifted = create_step_waveform(t, steps, step_times, shift)