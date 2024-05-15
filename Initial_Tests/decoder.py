import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataframe1 = pd.read_csv('Initial_Tests/to_decode.csv', header = None)
to_decode = dataframe1.to_numpy()

plt.plot(to_decode)
plt.show()

dataframe2 = pd.read_csv('Initial_Tests/channel_response_short.csv', header = None)
channel = dataframe2.to_numpy()

to_decode_freq = np.fft.fft(np.reshape(to_decode, 1024))

channel = np.reshape(channel, 300)
channel = np.pad(channel, (0, 1024 - len(channel)))
channel_freq = np.fft.fft(channel)

decoded = to_decode_freq / channel_freq

plt.plot(decoded)
plt.show()

sent = np.fft.ifft(decoded)
plt.plot(sent)
plt.show()