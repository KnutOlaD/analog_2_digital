'''
Idealized aliasing example that generates the 24-hour sampled data of 
methane concentration at in bottom water and how it is modulated
by the K1 tide.

author: Knut Ola Dølven
'''

import numpy as np
import matplotlib.pyplot as plt

#set plotting style to dark
plt.style.use('dark_background')

def generate_methane_signal(sampling_frequency=1/(24*3600),  # 1 sample every day
                            duration=300*3600*24, # 1 year in seconds
                            pressure_data = None,
                            noise_level=5):
    '''
    Generates methane data obtained with a certain frequency rate and noise addition

    Parameters:
    sampling_frequency (float): Frequency of the methane signal in cycles per seconds (sampling rate).
    pressure_data (numpy.ndarray): Optional pressure data in values per sampling instance,
    to modulate the methane signal. If None, a default K1 tide pressure signal is generated.
    duration (float): Duration of the signal in seconds.
    noise_level (float): Standard deviation of the noise to be added to the signal.

    Returns:
    t: Time vector in seconds.
    methane_signal: Generated methane signal with noise.


    '''

    if pressure_data is None:
        K1_frequency = 1/(23.93447213*3600)  # K1 tide frequency in cycles per second
        M2_frequency = 1/(12.420601*3600)  # M2 tide frequency in cycles per second
        pressure_data = np.sin(2 * np.pi * K1_frequency * np.arange(80*24*3600, duration+80*24*3600, 1/(sampling_frequency)))  # Simulated pressure data

    t = np.arange(0, duration, 1/(sampling_frequency))
    methane_signal = 150+10*pressure_data + np.random.normal(0, noise_level, len(t))
    return t, methane_signal

def generate_pressure_data(sampling_frequency=1/24, # 1 sample every day
                           duration=300*3600*24,  # 1 year in seconds
                           tide_type='K1',
                           amplitude=1,
                           base_pressure=0):
    '''
    Generates pressure data for a given tide type.

    Parameters:
    sampling_frequency (float): Frequency of the pressure signal in cycles per seconds (sampling rate).
    duration (float): Duration of the signal in seconds.
    tide_type (str): Type of tide to generate ('K1' or 'M2').
    amplitude (float): Amplitude of the pressure signal.
    base_pressure (float): Base pressure value to be added to the signal.

    Returns:
    t: Time vector in seconds.
    pressure_data: Generated pressure data.
    '''

    if tide_type == 'K1':
        frequency = 1/(23.93447213*3600)  # K1 tide frequency in cycles per second
    elif tide_type == 'M2':
        frequency = 1/(12.420601*3600)  # M2 tide frequency in cycles per second
    else:
        raise ValueError("Unsupported tide type. Use 'K1' or 'M2'.")

    t = np.arange(0, duration, 1/(sampling_frequency))
    pressure_data = np.sin(2 * np.pi * frequency * t)* amplitude + base_pressure  # Simulated pressure data
    
    return t, pressure_data

# generate data
t, methane_signal = generate_methane_signal()
# do a simple 5th order polynomial fit to the data
#get the time vector into days
t_days = t / (24 * 3600)  # Convert time to days
# use something from the numpy or scipy library
from numpy import polyfit, poly1d
spline_coeffs = polyfit(t, methane_signal, 5)
spline = poly1d(spline_coeffs)
# plot the data
plt.figure(figsize=(12, 6))
plt.plot(t_days, methane_signal, label='Methane Signal',color='w')
# plot spline fit
#plt.plot(t_days, spline(t), linewidth = 5, label='Spline Fit', color='orange', linestyle='--')
plt.title('Obtained methane data', fontsize=16)
plt.xlabel('Days', fontsize=14)
plt.ylabel('Methane Concentration [nmol/L]', fontsize=14)
plt.xlim(0, np.max(t_days))  # Limit x-axis to one year  # Limit y-axis to reasonable range
plt.legend(fontsize=14)
# increase font sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# generate a 1 day dataset sampled at 1 sample per second
t_1day, methane_signal_1day = generate_methane_signal(sampling_frequency=1/3600, duration=24*3600*5)

plt.figure(figsize=(12, 6))

t_1day = t_1day / 3600  # Convert time to hours for better readability
plt.plot(t_1day, methane_signal_1day, label='Methane Signal (1 sample/sec)', color='w')
plt.title('Methane Signal Sampled at 1 Sample/hour', fontsize=16)

plt.xlabel('Hours', fontsize=14)
plt.ylabel('Methane Concentration [nmol/L]', fontsize=14)
plt.xlim(0, 24*5)  # Limit x-axis to one day
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# create a full dataset for 300 days sampled at 1 sample per hour
t_full, methane_signal_full = generate_methane_signal(sampling_frequency=1/3600, duration=300*24*3600)
plt.figure(figsize=(12, 6))
t_full_days = t_full / (24 * 3600)  # Convert time to days
plt.plot(t_full_days, methane_signal_full, label='Methane Signal (1 sample/hour)', color='w')
plt.title('Methane Signal Sampled at 1 Sample/hour for 300 Days', fontsize=16)
plt.xlabel('Days', fontsize=14)
plt.ylabel('Methane Concentration [nmol/L]', fontsize=14)
plt.xlim(0, 300)  # Limit x-axis to 300 days
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# geneerate some pressure data for K1 tide and plot it.
t_pressure, pressure_data = generate_pressure_data(sampling_frequency=1/3600, duration=300*24*3600, tide_type='K1', amplitude=2, base_pressure=100)
# add a bit of noise to the pressure data
pressure_data += np.random.normal(0, 0.5, len(t_pressure))
plt.figure(figsize=(12, 6))
plt.plot(t_pressure / (24 * 3600), pressure_data, label='Pressure data', color='w')
plt.title('Pressure Data Sampled at 1 Sample/hour', fontsize=16)
plt.xlabel('Days', fontsize=14)
plt.ylabel('Pressure [dbar]', fontsize=14)
plt.xlim(0, 300)  # Limit x-axis to 300 days
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# plot a 5 day period of pressure data and 5 day period of methane data in 
# a plot with a shared x-axis but different y-axes
plt.figure(figsize=(12, 6))
# Select a 5-day period for both pressure and methane data
start_day = 100  # Start at day 100
end_day = start_day + 5  # End at day 105
plt.plot(t_pressure[:-18] / (24 * 3600), pressure_data[18:], label='Pressure Data', color='w')
#add a second y-axis for the methane data
plt.twinx()  # Create a second y-axis
plt.plot(t_full_days, methane_signal_full, label='Methane Signal', color='orange')
plt.title('Pressure and Methane Data Over 5 Days', fontsize=16)
plt.xlabel('Days', fontsize=14)
plt.ylabel('Pressure [dbar]', fontsize=14)
plt.xlim(start_day, end_day)  # Limit x-axis to the 5-day
#plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()





# calculate the frequency resulting from an aliasing effect between
# the K1 tide and the sampling frequency
K1_frequency = 1/(23.93447213*3600)  # K
sampling_frequency = 1/(24*3600)  # Sampling frequency in cycles per second
aliasing_frequency = np.abs(K1_frequency - sampling_frequency)

print(f"Aliasing frequency: {aliasing_frequency:.10f} cycles per second")

# get the period in days
aliasing_period_days = 1 / aliasing_frequency / (24 * 3600)  # Convert to days
print(f"Aliasing period: {aliasing_period_days:.10f} days")

