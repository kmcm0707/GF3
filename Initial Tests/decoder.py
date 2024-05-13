import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataframe1 = pd.read_csv('recieved_file_1.csv', header = None)
to_decode = dataframe1.to_numpy()

dataframe2 = pd.read_csv('channel_response_short.csv', header = None)
channel = dataframe2.to_numpy()

to_decode_freq = np.fft.fft(np.reshape(to_decode, 1024))

channel = np.pad(channel, (0, 1024 - len(channel)), "constant")
channel_freq = np.fft.fft(np.reshape(channel, 300))

decoded = to_decode_freq / channel_freq
plt.plot(decoded)
plt.show()