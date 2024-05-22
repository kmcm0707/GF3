from synchronisation import *
# from sound_generator import *
import numpy as np
import scipy.signal
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import time
import pyaudio

'''
method to estimate channel coefficients through sending known pilot symbols and interpolating the missing freq and time 
(interpolation after sampling)

steps:
1. send pilot symbols of known time and freq of equal amplitudes 
2. upon receiving the symbols, synchronise to ensure correct start of symbols
3. point wise division to retrieve channel gains across pilot time and freq
4. interpolation across time and freq to fill in missing fields (can use ML, deep learning, plane fitting)

Note:
- freq from 1 to 16k
- each pilot symbols sent for 2s
- can be used to combat echo is spaced equally throughout the sent information symbols (accounts for changing channel across time and freq)
- accounts for amplitude attenuation
- taking an instance in discrete time, the channel coeff against freq should look like a rectangular waveform
  Upon DFT, it should give a channel response in discrete time that looks like a sinc

Question
- does not account for freq distortion? (Check with jossy)
- am I just sending sine waves of diff freq at diff time?
- is each pilot symbol sent just amplitude, is it possible to send phase and amplitude and check for phase shift (distortion) by the channel?
'''


'''
synchronisation to return start of time_data
'''
# duration = 5
# fs = 44100
# t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)

# y = pd.read_csv('sample_spectrums_2/sine1k.csv', header=None).to_numpy()
# y= np.reshape(y, len(y))
# plt.plot(y)
# plt.show()

# x = sin = np.sin(2 * np.pi * np.arange(fs * duration) * 1000 / fs).astype(np.float32)

# max_index, cross_correlation, lags = matched_filter_synchronisation(y,x, duration, fs)
# cross_correlation = np.reshape(cross_correlation, cross_correlation.shape[1])

# plt.plot(cross_correlation)
# plt.vlines(max_index, -cross_correlation[max_index], cross_correlation[max_index], colors='r')
# plt.show()
# correct_time_data = y[lags[max_index]:lags[max_index] + len(x)]
# print(len(x))
# print(len(correct_time_data))
# plt.plot(correct_time_data)
# plt.show()


'''
test if superimposing sine numpy arrays would transmit and receive different freq sine waves simultaneously
'''
def generate_sound(samples, volume, fs): # volume range [0.0, 1.0]
    p = pyaudio.PyAudio()
    time.sleep(1)
    output_bytes = (volume * samples).tobytes()
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    # play. May repeat with different volume values (if done interactively)
    start_time = time.time()
    stream.write(output_bytes)
    print("Played sound for {:.2f} seconds".format(time.time() - start_time))
    stream.stop_stream()
    stream.close()
    p.terminate()

# fs = 44100
# duration = 10.0  # in seconds, may be float

# t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)
# sin1 = np.sin(2 * np.pi * np.arange(fs * duration) * 1000 / fs).astype(np.float32)
# sin2 = np.sin(2 * np.pi * np.arange(fs * duration) * 3000 / fs).astype(np.float32)
# superimpose = sin1 + sin2
# generate_sound(superimpose, 1, fs)


'''
assuming an info sequence of 500 x 1056 info symbols
if we want to send 10 instances of pilot symbols, send one pilot symbol every 50 info symbols 
need to implement sending two pilot symbols at the same time, is this just summing up the symbols of 2 diff sine symbols and generate sound using pyaudio
'''
fs = 44100
duration = 1  # in seconds, may be float

t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)
sin1 = np.sin(2 * np.pi * np.arange(fs * duration) * 1000 / fs).astype(np.float32)
sin4 = np.sin(2 * np.pi * np.arange(fs * duration) * 4000 / fs).astype(np.float32)
sin5 = np.sin(2 * np.pi * np.arange(fs * duration) * 5000 / fs).astype(np.float32)
sin8 = np.sin(2 * np.pi * np.arange(fs * duration) * 8000 / fs).astype(np.float32)
sin9 = np.sin(2 * np.pi * np.arange(fs * duration) * 9000 / fs).astype(np.float32)
sin12 = np.sin(2 * np.pi * np.arange(fs * duration) * 12000 / fs).astype(np.float32)
sin13 = np.sin(2 * np.pi * np.arange(fs * duration) * 13000 / fs).astype(np.float32)
sin16 = np.sin(2 * np.pi * np.arange(fs * duration) * 16000 / fs).astype(np.float32)
pilot_state1 = sin1 + sin9 + sin16
pilot_state2 = sin5 + sin13
pilot_state3 = sin1 + sin8 + sin16
pilot_state4 = sin4 + sin12

pilot_states = [pilot_state1, pilot_state2, pilot_state3, pilot_state4, pilot_state1, pilot_state2, pilot_state3, pilot_state4, pilot_state1, pilot_state2]
info_sequence = np.zeros(shape=(1000, 1056))
for x in reversed(range(0, 10)):
    pos = int(x*(1000/10))
    info_sequence = np.insert(info_sequence, pos, pilot_state1)

print(info_sequence)
print(len(info_sequence))

# generate_sound(info_sequence, 1, fs)
# sum bug going on, sound generated is longer than expected and each pilot state is played for more than 1s

'''
after receiving the audio file, DFT received and DFT transmitted, divide and get channel coeff
carry out interpolation to get all the channel coeff
'''