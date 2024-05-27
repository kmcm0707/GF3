import numpy as np
import pandas as pd
def sudo_random_generator(seed, length):
    np.random.seed(seed)
    return np.random.randint(0, 4, size = length)

if __name__ == "__main__":
    seed = 0
    length = 2047
    block = sudo_random_generator(seed, length)
    block = np.array(block)
    print(block.shape)
    pd.DataFrame(block).to_csv('OFDM_block_ranints.csv', index = False, header = False)
    bits = []
    for i in block:
        bits.append(np.binary_repr(i, width = 2))
    bits = np.array(bits)
    print(bits)
    pd.DataFrame(bits).to_csv('OFDM_block_ranbits.csv', index = False, header = False)

