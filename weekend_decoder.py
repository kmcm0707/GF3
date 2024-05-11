### WEEKEND DECODER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load file1.csv and channel.csv as vectors

dataframe1 = pd.read_csv('file1.csv', header=None)
to_decode = dataframe1.to_numpy()

dataframe2 = pd.read_csv('channel.csv', header=None)
channel = dataframe2.to_numpy()

# Split into 1056 bit length OFDM 'symbols'

symbol_len = 1056
no_symbols = int(len(to_decode) / symbol_len)
symbols = np.split(to_decode, len(to_decode)/symbol_len)

# Remove first 32 elements of each 'symbol' (cyclic prefix)

for index, i in enumerate(symbols):
    symbols[index] = i[32:]

# DFT of each 'symbol' (...should be complex)
symbols_freq = np.ones((950, 1024))
symbols_freq = symbols_freq.astype(complex)
for index, i in enumerate(symbols):
    temp = np.reshape(i, 1024)
    symbols_freq[index] = np.fft.fft(temp)
"""
print(symbols_freq.shape)
print(to_decode.shape)
to_decode = np.delete(to_decode, 1)
print(to_decode.shape)
to_decode_freq = np.fft.fft(to_decode)
print(to_decode_freq.shape)
symbols_freq_2 = np.split(to_decode_freq, len(to_decode_freq)/1056)"""

"""for index, i in enumerate(symbols_freq_2):
    symbols_freq_2[index] = i[31:]"""



assert np.round(symbols_freq[0][1], 5) == np.round(np.conjugate(symbols_freq[0][-1]), 5)

# Divide by DFT of Channel Impulse Response (??)

channel = np.reshape(channel, 30)

channel = np.pad(channel, (0, 1024 - len(channel)), "constant")

#channel_2 = 
channel_freq = np.fft.fft(channel)

print(symbols_freq[0][0])
print(symbols_freq[0][1])
print(symbols_freq[0][1023])
recieved_freq = symbols_freq / channel_freq

# For each element, find closest conselation symbol (√2 + √2 j, √2 - √2 j, -√2 - √2 j, - √2 + √2j)

constellations = np.zeros((1, 511))

for index, i in enumerate(recieved_freq):
    if index == 0:
        print(i[0])
    constellations = np.vstack((constellations,i[1:512]))

constellations = np.delete(constellations, 0, 0)
# Match each conselation symbol to the bits (Gray code)
print(constellations.shape)
print(constellations[0][510])
binary = []

plt.scatter(constellations[4][:100].real, constellations[4][:100].imag)
plt.scatter(constellations[4][100:250].real, constellations[4][100:250].imag, color='red')
plt.scatter(constellations[4][250:].real, constellations[4][250:].imag, color='green')
plt.show()

# Hmmm?? The lower frequencies seem to be correct, but the higher frequencies don't seem to work

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

