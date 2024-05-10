### WEEKEND DECODER

import numpy as np
import pandas as pd

# Load file1.csv and channel.csv as vectors

dataframe1 = pd.read_csv('file1.csv', header=None)
to_decode = dataframe1.to_numpy()

dataframe2 = pd.read_csv('channel.csv', header=None)
channel = dataframe2.to_numpy()

# Split into 1056 bit length OFDM 'symbols'

symbol_len = 1056

symbols = np.split(to_decode, len(to_decode)/symbol_len)


# Remove first 32 elements of each 'symbol' (cyclic prefix)

for index, i in enumerate(symbols):
    symbols[index] = i[31:]

# DFT of each 'symbol' (...should be complex)
symbols_freq = np.ones((950, 1024))
symbols_freq = symbols_freq.astype(complex)

for index, i in enumerate(symbols):
    temp = np.delete(i, 1)
    symbols_freq[index] = np.fft.fft(temp)


assert np.round(symbols_freq[0][1], 5) == np.round(np.conjugate(symbols_freq[0][-1]), 5)

# Divide by DFT of Channel Impulse Response (??)

channel = np.delete(channel, 1)

channel = np.pad(channel, (0, 1024 - len(channel)), "constant")
channel_freq = np.fft.fft(channel)

recieved_freq = symbols_freq / channel_freq

# For each element, find closest conselation symbol (√2 + √2 j, √2 - √2 j, -√2 - √2 j, - √2 + √2j)

constellations = []

for index, i in enumerate(recieved_freq):
    constellations.append(i[1:512])

# Match each conselation symbol to the bits (Gray code)

binary = []

for symbol in constellations:
    for i in symbol:
        if np.real(i) >= 0 and np.imag(i) >= 0:
            binary.append("00")
        elif np.real(i) <= 0 and np.imag(i) >= 0:
            binary.append("01")
        elif np.real(i) <= 0 and np.imag(i) <= 0:
            binary.append("11")
        elif np.real(i) >= 0 and np.imag(i) <= 0:
            binary.append("10")
        else:
            print("uh oh!")
    
binary = ''.join(binary)
# output = binary.decode("ascii")

output = open("output.txt", "w")
output.write(binary)
output.close()

# Remove header + Convert to .txt (how, not sure!)

