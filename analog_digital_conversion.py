'''
Script to demonstrate the conversion of an analog signal to a digital signal and play around with 
the parameters of the conversion process. The script simulates an analog signal and applies various 
sampling and quantization techniques.

Author: Knut Ola Dølven
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#############
# FUNCTIONS #
#############

def generate_analog_signal(frequency=5, duration=1, sample_rate=10000):
    '''
    
    Generate "analog" signal (sine wave).
    
    Parameters:
    frequency (float): Frequency of the sine wave in Hz.
    duration (float): Duration of the signal in seconds.
    sample_rate (int): Number of samples per second.
    
    Returns:
    t (numpy.ndarray): Time vector.
    signal (numpy.ndarray): Generated analog signal.
    
    '''

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

def sample_signal(signal, sample_rate, original_rate):
    '''
    Sample the analog signal at a specified sample rate.
    
    Parameters:
    signal (numpy.ndarray): The analog signal to be sampled.
    sample_rate (int): The desired sample rate in Hz.
    original_rate (int): The original sample rate of the signal in Hz.
    
    Returns:
    sampled_signal (numpy.ndarray): The sampled signal.
    t_sampled (numpy.ndarray): Time vector for the sampled signal.
    '''
    
    # Calculate the downsampling factor
    # It is done this way to ensure there are 
    downsample_factor = int(original_rate / sample_rate)
    
    # Sample the signal
    sampled_signal = signal[::downsample_factor]
    
    # Create a time vector for the sampled signal
    t_sampled = np.linspace(0, len(sampled_signal) / sample_rate, len(sampled_signal), endpoint=False)
    
    return t_sampled, sampled_signal

def quantize_signal(signal, num_bits):
    '''
    Quantize the sampled signal to a specified number of bits.
    
    Parameters:
    signal (numpy.ndarray): The sampled signal to be quantized.
    num_bits (int): Number of bits for quantization.
    
    Returns:
    quantized_signal (numpy.ndarray): The quantized signal.
    '''
    
    # Calculate the quantization levels
    levels = 2 ** num_bits
    
    # Find the minimum and maximum values of the signal
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # Scale the signal to the range of quantization levels
    scaled_signal = (signal - min_val) / (max_val - min_val) * (levels - 1)
    
    # Quantize the signal
    quantized_signal = np.round(scaled_signal).astype(int)

    # Scale back to the original range
    quantized_signal = quantized_signal / (levels - 1) * (max_val - min_val) + min_val
    
    return quantized_signal


if __name__ == "__main__":
    
    # Test the functions with example parameters

    #  Parameters
    duration = 1.0  # seconds
    frequency = 5.0  # Hz
    sample_rate = 20  # Hz
    num_bits = 3

    # Generate the analog signal
    t, signal = generate_analog_signal(duration, frequency, sample_rate)

    # Sample the signal
    t_sampled, sampled_signal = sample_signal(signal, sample_rate, sample_rate)

    # Quantize the signal
    quantized_signal = quantize_signal(sampled_signal, num_bits)

    # Plot the signals
    plt.figure(figsize=(12, 8))
    plt.plot(t, signal, label='Analog Signal', color='blue')
    plt.stem(t_sampled, sampled_signal, label='Sampled Signal', linefmt='orange', markerfmt='ro', basefmt=' ')
    plt.stem(t_sampled, quantized_signal, label='Quantized Signal', linefmt='green', markerfmt='go', basefmt=' ')
    plt.title('Analog, Sampled, and Quantized Signals')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()




