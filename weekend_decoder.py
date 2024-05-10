### WEEKEND DECODER

import numpy

# Load file1.csv and channel.csv as vectors

file1 = open('file1.csv')
to_decode = file1.split(",\n")

channel_file = open('channel.csv')
channel = channel_file.split(",\n")

# Split into 1056 bit length OFDM 'symbols'

symbol_len = 1056

symbols = []

# Remove first 32 elements of each 'symbol'

# DFT of each 'symbol' (...should be complex)

# Divide by DFT of Channel Impulse Response (??)

# For each element, find closest conselation symbol (√2 + √2 j, √2 - √2 j, -√2 - √2 j, - √2 + √2j)

# Match each conselation symbol to the bits (Gray code)

# Remove header + Convert to .txt (how, not sure!)

