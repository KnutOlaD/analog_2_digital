'''
# Create .wav files from generated "analog" signals and play them back using 


'''

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

sample_rate = 44100 # Hz
sound_duration = 5  # seconds

##################
# generate noise #
##################

data = np.random.uniform(-1, 1, sample_rate*sound_duration) # 10 second worth of random samples between -1 and 1
scaled = np.int16(data / np.max(np.abs(data)) * 32767) #multiply by 32767 to scale to int16 range

# create noise sound file and store it in a .wav file located in the current directory
sio.wavfile.write('noise.wav', sample_rate, scaled)

# play the sound file using the system's default audio player
#os.startfile('noise.wav')

######################
# generate sine tone #
######################

#Generate a sine wave sampled in 44200 Hz at 16 bits for 5 seconds duration.
#This is essentially a cd quality A4 tone.
frequency = 440  # (A4 note)
t = np.linspace(0, sound_duration, sample_rate * sound_duration, endpoint=False)
sine_wave = np.sin(2 * np.pi * frequency * t)  # Generate the sine wave
scaled_sine = np.int16(sine_wave / np.max(np.abs(sine_wave)) * 32767)  # Scale to int16 range to avoid clipping
# create sine wave sound file and store it in a .wav file located in the current directory
sio.wavfile.write('sine.wav', sample_rate, scaled_sine)
# play the sine wave sound file using the system's default audio player
#os.startfile('sine.wav')

# make a plot of the sine wave and the frequency spectrum of the sine wave
plt.figure(figsize=(12, 6))
plt.plot(t, sine_wave)
plt.title('Sine Wave Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

# Plot the frequency spectrum of the sine wave
plt.figure(figsize=(12, 6))
plt.magnitude_spectrum(sine_wave, Fs=sample_rate, scale='dB', color='C1')
plt.title('Frequency Spectrum of Sine Wave')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.show()

#Make the same plot but zoom in on the first 0.1 seconds of the sine wave
plt.figure(figsize=(12, 6))
plt.plot(t[:int(sample_rate * 0.1)], sine_wave[:int(sample_rate * 0.1)])
plt.title('Sine Wave Signal (Zoomed In)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(0, 0.1)
plt.show()

#further zoom
plt.figure(figsize=(12, 6))
plt.plot(t[:int(sample_rate * 0.01)], sine_wave[:int(sample_rate * 0.01)])
plt.title('Sine Wave Signal (Zoomed In Further)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(0, 0.01)
plt.show()

################################################
# Sample the sine wave at various sample rates
# and see what happens  #
################################################

#Sampling at below the Shannon-Nyquist rate (2*frequency), at the Shannon-Nyquist
#rate -1, exactly at 2*frequency and at the engineering sample rate.
sample_rates = [500, int(2*frequency-1), int(2*frequency), int(2.5*frequency)]
field_names = ['below_nyquist', 'nyquist_minus_1', 'nyquist', 'engineering_sample_rate']
sampled_data_dict = {}

for sr in sample_rates:
    # Resample the sine wave to the new sample rate
    t_resampled = np.linspace(0, sound_duration, sr * sound_duration, endpoint=False)
    sine_wave_resampled = np.sin(2 * np.pi * frequency * t_resampled)

    sampled_data_dict[field_names[sample_rates.index(sr)]] = {
        'sample_rate': sr,
        't': t_resampled,
        'sine_wave': sine_wave_resampled
    }

    # Plot the resampled sine wave
    plt.figure(figsize=(12, 6))
    plt.plot(t_resampled, sine_wave_resampled)
    plt.title(f'Sine Wave Signal at {sr} Hz')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.xlim(0, 0.1)
    plt.show()

    # create and save wav file for the resampled sine wave
    # scaled_resampled = np.int16(sine_wave_resampled / np.max(np.abs(sine_wave_resampled)) * 32767)
    # filename = f'sine_resampled_{sr}.wav'
    # sio.wavfile.write(filename, sr, scaled_resampled)
