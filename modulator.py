import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import pyaudio
import scipy.signal as signal


# x = np.linspace(-1/1024*2*np.pi, 2*np.pi, 1024)
# cosine = np.cos(x)
# cosine[0] = 0
# cosine[int(1024/2)] = 0
# # plt.plot(x, cosine)
# # plt.show()
# print(cosine[0])
# print(cosine[1])
# print(cosine[1023])


data = np.zeros(1024)
data[100:int(1024/2)-59] = 0.5
data[int(1024/2)+60:1024-99] = 0.5
data[int(1024/2)-59:int(1024/2)+60] = 1
data[int(1024/2)] = 0



# plt.plot(data)
# plt.show()


time_data = np.fft.ifft(data)
time_data = np.tile(time_data, 100)
# print(time_data.tolist())



p = pyaudio.PyAudio()
time.sleep(1)

volume = 1  # range [0.0, 1.0]
fs = 44100  # sampling rate, Hz, must be integer
duration = 1000.0  # in seconds, may be float
f = 4000.0  # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
# dataframe1 = pd.read_csv('weekend_files/file1.csv', header=None)
# to_decode = dataframe1.to_numpy()
# samples = np.reshape(to_decode, len(to_decode)).astype(np.float32)
# #samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

# generate chirp signal
#t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)
#samples = signal.chirp(t, f0=0, f1=16000, t1=int(duration), method='linear').astype(np.float32)
# per @yahweh comment explicitly convert to bytes sequence
print(time_data)
output_bytes = (volume * time_data).astype(np.float32).tobytes()


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


####################



