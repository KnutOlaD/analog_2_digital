'''
Script to demonstrate the effect of sampling an analog signal at various sample rates.

Author: Knut Ola Dølven

'''

import numpy as np
import matplotlib.pyplot as plt
import analog_digital_conversion as adc
import scipy.io as sio

###########################
# GENERATE ANALOG SIGNALS #
###########################

# SIGNAL 1, sine with frequency 440 Hz (A4 note)
frequency = 440  # Frequency of the sine wave in Hz
duration = 4  # Duration of the signal in seconds
sample_rate_analog = 44100  # Original sample rate in Hz
t, signal1 = adc.generate_analog_signal(frequency, duration, sample_rate_analog)

# SIGNAL 3, sine with frequency corresponding to a C5 note (523.25 Hz)
frequency2 = 523.25  # Frequency of the sine wave in Hz
t3, signal2 = adc.generate_analog_signal(frequency2, duration, sample_rate_analog)

# SIGNAL 2, sine with frequency corresponding to a E5 note (659.25 Hz)
frequency3 = 659.25  # Frequency of the sine wave in Hz
t2, signal3 = adc.generate_analog_signal(frequency3, duration, sample_rate_analog)

# SIGNAL 4, sine with frequency corresponding to a G5 note (783.99 Hz)
frequency4 = 783.99  # Frequency of the sine wave in Hz
t4, signal4 = adc.generate_analog_signal(frequency4, duration, sample_rate_analog)

# Add signals together to form more complex signal
signal12 = signal1 + signal2  # sum of signal 1 and signal 2
# add the third signal to the sum of the first two signals
signal123 = signal12 + signal3
# and adding the fourth signal
signal1234 = signal123 + signal4
# make a signal with signal 1,2, and 4
signal124 = signal1 + signal2 + signal4

###############################################
# PLOT FIGURES OF SIGNALS AND SUMS OF SIGNALS #
###############################################

# set plotting style to dark
plt.style.use('dark_background')

# PLOT SIGNALS ....

# plot simple sine wave signal 1
adc.plot_signal(t, signal1, title='"Analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)')
# plot signal but zoomed in on the first 0.01 seconds
adc.plot_signal(signal1[:int(0.01 * sample_rate_analog)], t[:int(0.01 * sample_rate_analog)], title='"Analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)', xlim=(0, 0.01), color='C0')
# plot the second signal ..
adc.plot_signal(signal2[:int(0.01 * sample_rate_analog)], t3[:int(0.01 * sample_rate_analog)], title='"Analogt" signal (sinusfunksjon) med frekvens 523.25 Hz (C5 note)', xlim=(0, 0.01), color='C1')
# ...third signal
adc.plot_signal(signal3[:int(0.01 * sample_rate_analog)], t2[:int(0.01 * sample_rate_analog)], title='"Analogt" signal (sinusfunksjon) med frekvens 659.25 Hz (E5 note)', xlim=(0, 0.01), color='C3')
# ...and the fourth signal
adc.plot_signal(signal4[:int(0.01 * sample_rate_analog)], t4[:int(0.01 * sample_rate_analog)], title='"Analogt" signal (sinusfunksjon) med frekvens 783.99 Hz (G5 note)', xlim=(0, 0.01), color='C2')
# plot signal 1 and 2 together in the same plot
two_signals = np.vstack((signal1[:int(0.01 * sample_rate_analog)], signal2[:int(0.01 * sample_rate_analog)]))
adc.plot_signal(two_signals.T, t[:int(0.01 * sample_rate_analog)], title='Signal 1 (A4) og Signal 2 (C5)', xlim=(0, 0.01),legend ='lower right')
# plot the sum of signal 1 and signal 2
adc.plot_signal(signal12[:int(0.01 * sample_rate_analog)], t[:int(0.01 * sample_rate_analog)], title='Signal 1 (A4) + Signal 2 (C5)', xlim=(0, 0.01), color='C4')
# plot the signal 1, 2, and 4 together in the same plot
three_signals = np.vstack((signal1[:int(0.01 * sample_rate_analog)], signal2[:int(0.01 * sample_rate_analog)], signal4[:int(0.01 * sample_rate_analog)]))
adc.plot_signal(three_signals.T, t[:int(0.01 * sample_rate_analog)], title='Signal 1 (A4), Signal 2 (C5) og Signal 4 (G5)', xlim=(0, 0.01), legend='lower right')  
# plot signal124
adc.plot_signal(signal124[:int(0.01 * sample_rate_analog)], t[:int(0.01 * sample_rate_analog)], title='Signal 1 (A4) + Signal 2 (C5) + Signal 4 (G5)', xlim=(0, 0.01), color='C6')    

# CREATE SOUNDFILES FOR THE SIGNALS

# Scale signals to create sound files for the three elementary signals and the complex signal
scaled_signal1 = np.int16(signal1 / np.max(np.abs(signal1)) * 32767)
scaled_signal2 = np.int16(signal2 / np.max(np.abs(signal2)) * 32767)
scaled_signal3 = np.int16(signal3 / np.max(np.abs(signal3)) * 32767)
scaled_signal123 = np.int16(signal123 / np.max(np.abs(signal123)) * 32767)
scaled_signal4 = np.int16(signal4 / np.max(np.abs(signal4)) * 32767)
scaled_signal124 = np.int16(signal124 / np.max(np.abs(signal124)) * 32767)
scaled_signal1234 = np.int16(signal1234 / np.max(np.abs(signal1234)) * 32767)

# Create .wav files for the signals
sio.wavfile.write('signal1.wav', sample_rate_analog, scaled_signal1)
sio.wavfile.write('signal2.wav', sample_rate_analog, scaled_signal2)
sio.wavfile.write('signal3.wav', sample_rate_analog, scaled_signal3)
sio.wavfile.write('signal123.wav', sample_rate_analog, scaled_signal123)
sio.wavfile.write('signal4.wav', sample_rate_analog, scaled_signal4)
sio.wavfile.write('signal1234.wav', sample_rate_analog, scaled_signal1234)
sio.wavfile.write('signal124.wav', sample_rate_analog, scaled_signal124)

# --- ########################################## --- #

###################################
# PLAY AROUND WITH SAMPLING RATES #
###################################

# Define different sample rates to illustrate the effect of sampling,
# we define a sample rate of 
# 220 Hz (too low) 
# 440Hz (at the lowest frequency of the signal, 440 Hz),
# 879 Hz (just below the Nyquist frequency), 
# 880 Hz (at the Nyquist frequency),
# 1000 Hz (above the Nyquist frequency for the 440 Hz signal),
# signal4*2 (Nyquist frequency for the highest pitched signal, 783.99 Hz),
# and signal4*2.5 Hz (the engineers nyqist frequency).

sample_rates = [220, 440, 800, 879, 880, 881, 1000, int(frequency4 * 2), int(frequency4 * 2.5), 11025]

# loop over all sample rates and sample the signal 1 and plot the results
for sample_rate in sample_rates:
    t_sampled = np.arange(0, duration, 1/sample_rate)
    t_sampled, sampled_signal = adc.sample_signal(t,signal1, sample_rate, sample_rate_analog)
    adc.plot_signal_w_samples(signal1, t, sampled_signal, t_sampled,
                          title=f'Diskret sampling på {sample_rate} Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                          xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.01), color='C0', sample_color='orange')

t_sampled,sampled_signal = adc.sample_signal(t, signal1, 880, sample_rate_analog)
#make a plot of the 880 Hz sampled signal for the first 4 seconds
adc.plot_signal_w_samples(signal1[:int(1 * sample_rate_analog)], t[:int(1 * sample_rate_analog)],
                          sampled_signal[:int(1 * sample_rate)], t_sampled[:int(1 * sample_rate)],
                          title='Diskret sampling på 880 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                          xlabel='Sekund', ylabel='Amplitude', xlim=(0, 1), color='C0', sample_color='orange')

# and for the first 0.1 seconds
adc.plot_signal_w_samples(signal1[:int(0.1 * sample_rate_analog)], t[:int(0.1 * sample_rate_analog)],
                          sampled_signal[:int(0.1 * sample_rate)], t_sampled[:int(0.1 * sample_rate)],
                          title='Diskret sampling på 880 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                          xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.1), color='C0', sample_color='orange')

# for the first 0.05 seconds
adc.plot_signal_w_samples(signal1[:int(0.5 * sample_rate_analog)], t[:int(0.5 * sample_rate_analog)],
                          sampled_signal[:int(0.5 * sample_rate)], t_sampled[:int(0.5 * sample_rate)],
                          title='Diskret sampling på 880 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                          xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.5), color='C0', sample_color='orange')

# and for 4 seconds
adc.plot_signal_w_samples(signal1, t, sampled_signal, t_sampled,
                          title='Diskret sampling på 880 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                          xlabel='Sekund', ylabel='Amplitude', xlim=(0, 4), color='C0', sample_color='orange')

# plot 0.1 seconds of 800 Hz sampled signal
t_sampled_800, sampled_signal_800 = adc.sample_signal(t, signal1, 800, sample_rate_analog)
adc.plot_signal_w_samples(signal1[:int(0.05 * sample_rate_analog)], t[:int(0.05 * sample_rate_analog)],
                            sampled_signal_800[:int(0.05 * 800)], t_sampled_800[:int(0.05 * 800)],
                            title='Diskret sampling på 800 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                            xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.05), color='C0', sample_color='orange')
# Try at 220 Hz
t_sampled_220, sampled_signal_220 = adc.sample_signal(t, signal1, 220, sample_rate_analog)
adc.plot_signal_w_samples(signal1[:int(0.05 * sample_rate_analog)], t[:int(0.05 * sample_rate_analog)],
                          sampled_signal_220[:int(0.05 * 220)], t_sampled_220[:int(0.05 * 220)],
                          title='Diskret sampling på 220 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                          xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.05), color='C0', sample_color='orange')

# and 1 second of 220 Hz sampled signal
adc.plot_signal_w_samples(signal1[:int(1 * sample_rate_analog)], t[:int(1 * sample_rate_analog)],
                           sampled_signal_220[:int(1 * 220)], t_sampled_220[:int(1 * 220)],
                           title='Diskret sampling på 220 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                           xlabel='Sekund', ylabel='Amplitude', xlim=(0, 1), color='C0', sample_color='orange')

# and for 0.01 seconds
adc.plot_signal_w_samples(signal1[:int(0.01 * sample_rate_analog)], t[:int(0.01 * sample_rate_analog)],     
                            sampled_signal_220[:int(0.01 * 220)], t_sampled_220[:int(0.01 * 220)],
                            title='Diskret sampling på 220 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                            xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.01), color='C0', sample_color='orange')

# try at 881 Hz (just below the Nyquist frequency)
t_sampled_881, sampled_signal_881 = adc.sample_signal(t, signal1, 881, sample_rate_analog)
adc.plot_signal_w_samples(signal1[:int(0.05 * sample_rate_analog)], t[:int(0.05 * sample_rate_analog)], 
                            sampled_signal_881[:int(0.05 * 881)], t_sampled_881[:int(0.05 * 881)],
                            title='Diskret sampling på 881 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                            xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.05), color='C0', sample_color='orange')

# try with 100 Hz
t_sampled_100, sampled_signal_100 = adc.sample_signal(t, signal1, 100, sample_rate_analog)  
adc.plot_signal_w_samples(signal1[:int(0.05 * sample_rate_analog)], t[:int(0.05 * sample_rate_analog)], 
                            sampled_signal_100[:int(0.05 * 100)], t_sampled_100[:int(0.05 * 100)],
                            title='Diskret sampling på 100 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                            xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.05), color='C0', sample_color='orange')

# plot a sinusoid wave that fits the sampled signal at 100 Hz
adc.plot_signal_w_samples(signal1[:int(0.05 * sample_rate_analog)], t[:int(0.05 * sample_rate_analog)],
                           sampled_signal_100[:int(0.05 * 100)], t_sampled_100[:int(0.05 * 100)],
                           title='Diskret sampling på 100 Hz av "analogt" signal (sinusfunksjon) med frekvens 440 Hz (A4 note)',
                           xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.05), color='C0', sample_color='orange')




# get the frequency of the aliased signal
frequency_aliased = 100 - frequency % 100
frequency_aliased = 40
#plot the aliased signal, i.e. a sinusoid wave with the frequency of the aliased signal
# create an aliased signal vector with dense time vector
t_aliased = np.arange(0, duration, 1/sample_rate_analog)
alias_signal = np.sin(2 * np.pi * frequency_aliased * t_aliased)
# plot the aliased signal on top of the 440hz signal together with the sampling points
plt.figure(figsize=(12, 6))
plt.plot(t_aliased, alias_signal, label='Aliased Signal (40 Hz)', color='C5')
plt.stem(t_sampled_100, sampled_signal_100, label='Sampled Signal at 100 Hz', linefmt='orange', markerfmt='ro', basefmt=' ')
plt.plot(t, signal1, label='Original Signal (440 Hz)', color='C0')
plt.axhline(0, color='white', linestyle='-', linewidth=1)
plt.title('Aliased Signal at 100 Hz vs Original Signal (440 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, 0.05)
plt.legend(loc='upper right')
plt.show()
# plot the 444 hz signal, the stemps for the sampled signal, and the aliased signal on top
plt.figure(figsize=(12, 6))


# CREATE SOUNDFILES FOR THE SAMPLED SIGNALS

# create one for 220 Hz, 880 Hz, 1000 Hz, and 12000 Hz
t_sampled_220 = np.arange(0, duration, 1/220)
sampled_signal_220 = adc.sample_signal(t, signal1, 220, sample_rate_analog)[1]
scaled_sampled_signal_220 = np.int16(sampled_signal_220 / np.max(np.abs(sampled_signal_220)) * 32767)
sio.wavfile.write('sampled_signal_220.wav', 220, scaled_sampled_signal_220)
t_sampled_880 = np.arange(0, duration, 1/880)
sampled_signal_880 = adc.sample_signal(t, signal1, 880, sample_rate_analog)[1]
scaled_sampled_signal_880 = np.int16(sampled_signal_880 / np.max(np.abs(sampled_signal_880)) * 32767)
sio.wavfile.write('sampled_signal_880.wav', 880, scaled_sampled_signal_880)
t_sampled_1000 = np.arange(0, duration, 1/1000)
sampled_signal_1000 = adc.sample_signal(t, signal1, 1000, sample_rate_analog)[1]
scaled_sampled_signal_1000 = np.int16(sampled_signal_1000 / np.max(np.abs(sampled_signal_1000)) * 32767)
sio.wavfile.write('sampled_signal_1000.wav', 1000, scaled_sampled_signal_1000)
t_sampled_12000 = np.arange(0, duration, 1/10000)
sampled_signal_11025 = adc.sample_signal(t, signal1, 11025, sample_rate_analog)[1]
scaled_sampled_signal_11025 = np.int16(sampled_signal_11025 / np.max(np.abs(sampled_signal_11025)) * 32767)
sio.wavfile.write('sampled_signal_11025.wav', 11025, scaled_sampled_signal_11025)

# DO THE SAME EXERCISE FOR THE COMPLEX SIGNAL (signal124)

# loop over all sample rates and sample the complex signal (signal124) and plot the results
for sample_rate in sample_rates:
    t_sampled = np.arange(0, duration, 1/sample_rate)
    t_sampled, sampled_signal = adc.sample_signal(t, signal124, sample_rate, sample_rate_analog)
    adc.plot_signal_w_samples(signal124, t, sampled_signal, t_sampled,
                          title=f'Diskret sampling på {sample_rate} Hz av "analogt" signal (signal124)',
                          xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.01), color='C6', sample_color='orange')
    
# CREATE SOUNDFILES FOR THE SAMPLED COMPLEX SIGNALS #
# create one for 220 Hz, 880 Hz, 1000 Hz, and 12000 Hz
t_sampled_220 = np.arange(0, duration, 1/220)
sampled_signal_220 = adc.sample_signal(t, signal124, 220, sample_rate_analog)[1]
scaled_sampled_signal_220 = np.int16(sampled_signal_220 / np.max(np.abs(sampled_signal_220)) * 32767)
sio.wavfile.write('sampled_signal124_220.wav', 220, scaled_sampled_signal_220)
t_sampled_880 = np.arange(0, duration, 1/880)   
sampled_signal_880 = adc.sample_signal(t, signal124, 880, sample_rate_analog)[1]
scaled_sampled_signal_880 = np.int16(sampled_signal_880 / np.max(np.abs(sampled_signal_880)) * 32767)
sio.wavfile.write('sampled_signal124_880.wav', 880, scaled_sampled_signal_880)
t_sampled_1000 = np.arange(0, duration, 1/1000)
sampled_signal_1000 = adc.sample_signal(t, signal124, 1000, sample_rate_analog)[1]
scaled_sampled_signal_1000 = np.int16(sampled_signal_1000 / np.max(np.abs(sampled_signal_1000)) * 32767)
sio.wavfile.write('sampled_signal124_1000.wav', 1000, scaled_sampled_signal_1000)
t_sampled_11025 = np.arange(0, duration, 1/11025)
sampled_signal_11025 = adc.sample_signal(t, signal124, 11025, sample_rate_analog)[1]
scaled_sampled_signal_11025 = np.int16(sampled_signal_11025 / np.max(np.abs(sampled_signal_11025)) * 32767)
sio.wavfile.write('sampled_signal124_11025.wav', 11025, scaled_sampled_signal_11025)  


# make one additional plot using the 2.5 * highest frequency as sample rate
sample_rate_2_5 = int(frequency4 * 2.5)
#sample_rate_2_5 = 2756
# Sample the complex signal (signal124) at the 2.5 * highest frequency
t_sampled_2_5,sampled_signal_2_5 = adc.sample_signal(t, signal124, sample_rate_2_5, sample_rate_analog)
adc.plot_signal_w_samples(signal124, t, sampled_signal_2_5, t_sampled_2_5,
                      title=f'Diskret sampling på {sample_rate_2_5} Hz av signal 1 + signal 2 + signal 3',
                      xlabel='Sekund', ylabel='Amplitude', xlim=(0, 0.01), color='C6', sample_color='orange')
# CREATE SOUNDFILE FOR THE SAMPLED COMPLEX SIGNAL AT 2.5 * HIGHEST FREQUENCY
scaled_sampled_signal_2_5 = np.int16(sampled_signal_2_5 / np.max(np.abs(sampled_signal_2_5)) * 32767)
sio.wavfile.write('sampled_signal124_2_5.wav', sample_rate_2_5, scaled_sampled_signal_2_5)

#################################################################################
# TO MAKE SOUND FILES WE NEED TO USE THE SINE FUNCTIONS, NOT THE DIGITAL SIGNAL #
#################################################################################

# Create sound files for the sine functions at the different sample rates
sio.wavfile.write('sine_signal_220.wav', 220, np.int16(np.sin(2 * np.pi * frequency * t_sampled_220) / np.max(np.abs(np.sin(2 * np.pi * frequency * t_sampled_220))) * 32767))
sio.wavfile.write('sine_signal_880.wav', 880, np.int16(np.sin(2 * np.pi * frequency * t_sampled_880) / np.max(np.abs(np.sin(2 * np.pi * frequency * t_sampled_880))) * 32767))
sio.wavfile.write('sine_signal_1000.wav', 1000, np.int16(np.sin(2 * np.pi * frequency * t_sampled_1000) / np.max(np.abs(np.sin(2 * np.pi * frequency * t_sampled_1000))) * 32767))
sio.wavfile.write('sine_signal_11025.wav', 11025, np.int16(np.sin(2 * np.pi * frequency * t_sampled_11025) / np.max(np.abs(np.sin(2 * np.pi * frequency * t_sampled_11025))) * 32767))
sio.wavfile.write('sine_signal_2_5.wav', sample_rate_2_5, np.int16(np.sin(2 * np.pi * frequency * t_sampled_2_5) / np.max(np.abs(np.sin(2 * np.pi * frequency * t_sampled_2_5))) * 32767))
# write one for 440 Hz too
sio.wavfile.write('sine_signal_440.wav', sample_rate_analog, np.int16(np.sin(2 * np.pi * frequency * t) / np.max(np.abs(np.sin(2 * np.pi * frequency * t))) * 32767))   
# and 800 Hz
sio.wavfile.write('sine_signal_800.wav', sample_rate_analog, np.int16(np.sin(2 * np.pi * frequency2 * t) / np.max(np.abs(np.sin(2 * np.pi * frequency2 * t))) * 32767)) 

# Create sound files for the complex signal (signal124) at the different sample rates
signal_124_220 = np.sin(2 * np.pi * frequency * t_sampled_220) + \
                  np.sin(2 * np.pi * frequency2 * t_sampled_220) + \
                  np.sin(2 * np.pi * frequency4 * t_sampled_220)

sio.wavfile.write('complex_signal_124_220.wav', 220, np.int16(signal_124_220 / np.max(np.abs(signal_124_220)) * 32767))
signal_124_880 = np.sin(2 * np.pi * frequency * t_sampled_880) + \
                  np.sin(2 * np.pi * frequency2 * t_sampled_880) + \
                  np.sin(2 * np.pi * frequency4 * t_sampled_880)

sio.wavfile.write('complex_signal_124_880.wav', 880, np.int16(signal_124_880 / np.max(np.abs(signal_124_880)) * 32767))
signal_124_1000 = np.sin(2 * np.pi * frequency * t_sampled_1000) + \
                   np.sin(2 * np.pi * frequency2 * t_sampled_1000) + \
                   np.sin(2 * np.pi * frequency4 * t_sampled_1000)

sio.wavfile.write('complex_signal_124_1000.wav', 1000, np.int16(signal_124_1000 / np.max(np.abs(signal_124_1000)) * 32767))
signal_124_11025 = np.sin(2 * np.pi * frequency * t_sampled_11025) + \
                    np.sin(2 * np.pi * frequency2 * t_sampled_11025) + \
                    np.sin(2 * np.pi * frequency4 * t_sampled_11025)

sio.wavfile.write('complex_signal_124_11025.wav', 11025, np.int16(signal_124_11025 / np.max(np.abs(signal_124_11025)) * 32767))
signal_124_2_5 = np.sin(2 * np.pi * frequency * t_sampled_2_5) + \
                   np.sin(2 * np.pi * frequency2 * t_sampled_2_5) + \
                   np.sin(2 * np.pi * frequency4 * t_sampled_2_5)

sio.wavfile.write('complex_signal_124_2_5.wav', sample_rate_2_5, np.int16(signal_124_2_5 / np.max(np.abs(signal_124_2_5)) * 32767)) 

# Create a complex signal with all four signals and plot 0.01 seconds of it
# with no ticks or labels on the axes
signal_all = signal1 + signal2 + signal3 + signal4

plt.figure(figsize=(6, 6))
plt.plot(t[:int(0.005 * sample_rate_analog)], signal_all[:int(0.005 * sample_rate_analog)], color='C5')
plt.xticks([])
plt.yticks([])
#remove the frame around the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
#plt.xlim(0, 0.005)  # Limit x-axis to the first 0.005 seconds
plt.show()

# make the same plot but with some low frequency noise to it
noise = np.random.normal(0, 0.25, len(signal_all[:int(0.005 * sample_rate_analog)]))
plt.figure(figsize=(6, 6))
plt.plot(t[:int(0.005 * sample_rate_analog)], signal_all[:int(0.005 * sample_rate_analog)] + noise, color='C6')
plt.xticks([])
plt.yticks([])
#remove the frame around the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.show()

# make the same plot but sample to show digitization and quantization
sample_rate = 5000  # Sample rate in Hz
#t_sampled = np.arange(0, duration, 1/sample_rate)
t_sampled,sampled_signal = adc.sample_signal(t, signal_all[:int(0.005 * sample_rate_analog)] + noise, sample_rate, sample_rate_analog)
# also, bin the noise to make it look more like a digital signal
sampled_signal = np.round(sampled_signal * 2) / 2  # Quantize to 1 decimal place

plt.figure(figsize=(6, 6))
#plt.plot(t[:int(0.005 * sample_rate_analog)], signal_all[:int(0.005 * sample_rate_analog)] + noise, color='C6')
plt.plot(t_sampled[:int(0.005 * sample_rate)+3], sampled_signal[:int(0.005 * sample_rate)+3], 'o', color='C7')
#plt.stem(t_sampled[:int(0.005 * sample_rate)+3], sampled_signal[:int(0.005 * sample_rate)+3], linefmt='w', markerfmt='C7o', basefmt=' ')
plt.axhline(0, color='white', linestyle='-', linewidth=1)
# plot faint dashed lines at every quantization level
for i in np.arange(-2.5, 4, 0.5):
    plt.axhline(i, color='C7', linestyle='--', linewidth=0.5, alpha=0.5)    
#plot faint dashed vertical lines at every sample point
for i in range(0, len(t_sampled[:int(0.005 * sample_rate)+3])):
    plt.axvline(t_sampled[i], color='C7', linestyle='--', linewidth=0.5, alpha=0.5)
plt.plot(t_sampled[:int(0.005 * sample_rate)+3], sampled_signal[:int(0.005 * sample_rate)+3], color='C7', linewidth=1)
plt.xticks([])
plt.yticks([])
#plt.xlim(0, 0.005)  # Limit x-axis to the first 0.005 seconds
#remove the frame around the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.show()

