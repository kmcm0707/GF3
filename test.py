import numpy as np
with open('ldpc_jossy\max_test_in.txt', 'rb') as f:
    bytes=f.read()
bits = np.unpackbits(np.frombuffer(bytes, dtype=np.uint8))
bits_string = ''.join(str(x) for x in bits)
print(bits_string)
