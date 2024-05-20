import numpy as np

import time
import pyaudio

# from sound_generator import *


with open('transmitter/data_file3.txt', 'r') as file:
    data = file.read().replace('\n', '')
    data = data.replace(' ', '')
    data = data.replace('.', '')

print(data)
print(type(bytearray(data, 'ascii')))

data_in_binary = '0'
data_in_binary += ('0'.join(format(ord(x), 'b') for x in data))
data_in_binary += ''.join(str(x) for x in ([0] * (1022 - len(data_in_binary) % 1022)))
data_in_binary = np.array(list(data_in_binary))

np.savetxt("foo.txt", data_in_binary, delimiter="", fmt='%s')
print(data_in_binary)
print(len(data_in_binary))
gray_mapping = np.split(data_in_binary, len(data_in_binary)/2)
#print(gray_mapping)
for index, x in enumerate(gray_mapping):
    if index == 0:
        print(x)
        assert (x == ['0', '1']).all()
    
    if (x == ['0', '0']).all():
        gray_mapping[index] = (1 + 1j)/np.sqrt(2)
    if (x == ['0', '1']).all():
        gray_mapping[index] = (-1 + 1j)/np.sqrt(2)
        #print((-1 + 1j)/np.sqrt(2))
    if (x == ['1', '1']).all():
        gray_mapping[index] = (-1 - 1j)/np.sqrt(2)
    if (x == ['1', '0']).all():
        gray_mapping[index] = (1 - 1j)/np.sqrt(2)
    



print(np.array(gray_mapping))
print(len(gray_mapping))
print(len(data_in_binary))
symbol = np.split(np.array(gray_mapping), len(gray_mapping)/511)

for index, x in enumerate(symbol):
    conj = np.conjugate(x)[::-1]
    symbol[index] = np.concatenate((x, conj), axis=None)
    symbol[index] = np.insert(symbol[index], 0, 0)
    symbol[index] = np.insert(symbol[index], 512, 0)

print(symbol[0])
info = np.fft.ifft(symbol) # should I iDFT the whole block or iDFT each 1024 symbol, is there a difference?

to_transmit = np.zeros(shape=(len(info), 1056))

for index, x in enumerate(info):
    cyclic_prefix = x[-32:]
    new_x = np.concatenate((cyclic_prefix, x), axis = None)
    to_transmit[index] = new_x


to_transmit = np.concatenate(to_transmit, axis=0)

# generate_sound(to_transmit, 1, sampling_freq = 44100)

fs = 44100 # Sampling frequency
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

to_transmit = to_transmit.astype(np.float32)
np.savetxt("foo.csv", to_transmit, delimiter="")

#generate_sound(to_transmit, 1, fs)