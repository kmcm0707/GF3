### WEEKEND DECODER

import numpy as np
import pandas as pd

# Load file1.csv and channel.csv as vectors

dataframe1 = pd.read_csv('file1.csv', header=None)
to_decode = dataframe1.to_numpy()

dataframe2 = pd.read_csv('file1.csv', header=None)
channel = dataframe2.to_numpy()

# Split into 1056 bit length OFDM 'symbols'

symbol_len = 1056

symbols = np.split(to_decode, symbol_len)

print(len(symbols[0]))


# Remove first 32 elements of each 'symbol' (cyclic prefix)

symbols = symbols[:][32:]

# for index, i in enumerate(symbols):
#     symbols[index] = i[31:]

print(len(symbols[0]))

# DFT of each 'symbol' (...should be complex)

# Divide by DFT of Channel Impulse Response (??)

# For each element, find closest conselation symbol (√2 + √2 j, √2 - √2 j, -√2 - √2 j, - √2 + √2j)

# Match each conselation symbol to the bits (Gray code)

# Remove header + Convert to .txt (how, not sure!)

