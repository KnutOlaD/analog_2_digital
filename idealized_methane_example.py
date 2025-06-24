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
plt.plot(t_days, methane_signal, label='Methane Signal',color='k')
# plot spline fit
plt.plot(t_days, spline(t), linewidth = 5, label='Spline Fit', color='orange', linestyle='--')
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
plt.plot(t_1day, methane_signal_1day, label='Methane Signal (1 sample/sec)', color='k')
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
plt.plot(t_full_days, methane_signal_full, label='Methane Signal (1 sample/hour)', color='k')
plt.title('Methane Signal Sampled at 1 Sample/hour for 300 Days', fontsize=16)
plt.xlabel('Days', fontsize=14)
plt.ylabel('Methane Concentration [nmol/L]', fontsize=14)
plt.xlim(0, 300)  # Limit x-axis to 300 days
plt.legend(fontsize=14)
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

