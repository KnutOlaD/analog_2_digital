'''
Script containing functions for playing around with analog to digital conversion. Contains functions
for 

generate_analog_signal: Creates an "analog" signal, i.e. this is really a digital signals with 
a high sample rate

sample_signal: Samples the analog signal at a specified sample rate. This function is also not 
strictly a sampling function but rather a downsampling function taking the original "analog" signal
and downsampling it such that the sample rate is equal to what specified by the user.

qantize_signal

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

def sample_signal(t_signal,signal, sample_rate, original_rate):
    '''
    Sample the analog signal at a specified sample rate.
    
    Parameters:
    t_signal (numpy.ndarray): Time vector for the original signal.
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
    t_sampled = t_signal[::downsample_factor]

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

def plot_signal(signal, t, title, xlabel='Time (s)', ylabel='Amplitude', xlim=None, ylim=None):
    """
    Helper function to plot a signal.
    
    Parameters:
    signal (numpy.ndarray): The signal to plot.
    t (numpy.ndarray): Time vector for the signal.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    xlim (tuple): Limits for the x-axis.
    ylim (tuple): Limits for the y-axis.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='Signal', color='C0')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid()
    plt.show()


def plot_signal(signal, 
                t, 
                title, 
                xlabel='Time (s)', 
                ylabel='Amplitude', 
                xlim=None, 
                ylim=None, 
                grid=False,
                color = 'C0',
                legend=None):
    """
    Helper function to plot a signal.
    
    Parameters:
    signal (numpy.ndarray): The signal to plot.
    t (numpy.ndarray): Time vector for the signal.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    xlim (tuple): Limits for the x-axis.
    ylim (tuple): Limits for the y-axis.
    grid (bool): Whether to show grid lines.
    """
    plt.figure(figsize=(12, 6))
    if np.ndim(signal) > 1:  # If signal is a matrix with two or more signals
        for i in range(np.shape(signal)[1]):
            plt.plot(t, signal[:, i], label=f'Signal {i+1}', color=f'C{i}')
    else:
        plt.plot(t, signal, label='Signal', color=color)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if grid == True:
        plt.grid()
    if legend is not None:
        plt.legend(fontsize=14, loc=legend)
    plt.show()

def plot_signal_w_samples(signal, t, sampled_signal, t_sampled, title, xlabel='Time (s)', ylabel='Amplitude', xlim=None, ylim=None, color='C0', sample_color='orange', grid=False):
    """
    Helper function to plot a signal with its sampled version.
    
    Parameters:
    signal (numpy.ndarray): The original signal to plot.
    t (numpy.ndarray): Time vector for the original signal.
    sampled_signal (numpy.ndarray): The sampled version of the signal.
    t_sampled (numpy.ndarray): Time vector for the sampled signal.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    xlim (tuple): Limits for the x-axis.
    ylim (tuple): Limits for the y-axis.
    color (str): Color for the original signal line.
    sample_color (str): Color for the sampled points.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='Analogt signal', color=color)
    plt.stem(t_sampled, sampled_signal, label='Samplet signal', linefmt=sample_color, markerfmt='ro', basefmt=' ')
    plt.axhline(0, color='white', linestyle='-', linewidth=1)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend(loc='lower right', fontsize=14)
    if grid:
        plt.grid()
    plt.show()


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




