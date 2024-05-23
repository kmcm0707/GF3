import numpy as np

import time
import pyaudio
import scipy
import pandas as pd

# from sound_generator import *


with open('transmitter/data_file.txt', 'r') as file:
    data = file.read().replace('\n', '')
    #data = data.replace(' ', '')
    #data = data.replace('.', '')

print(data)
print(type(bytearray(data, 'ascii')))

data_in_binary = ''
data_in_binary += (''.join(format(ord(x), 'b').zfill(8) for x in data))
data_in_binary += ''.join(str(x) for x in ([0] * (1022 - len(data_in_binary) % 1022)))
data_in_binary = np.array(list(data_in_binary))

np.savetxt("bee_data.txt", data_in_binary, delimiter="", fmt='%s')
print(data_in_binary)
print(len(data_in_binary))
gray_mapping = np.split(data_in_binary, len(data_in_binary)/2)
#print(gray_mapping)
constelatons = []
for index, x in enumerate(gray_mapping):
    if index == 0:
        print(x)
        assert (x == ['0', '1']).all()
    
    if (x == ['0', '0']).all():
        gray_mapping[index] = (1 + 1j)/np.sqrt(2)
        constelatons.append('A')
    if (x == ['0', '1']).all():
        gray_mapping[index] = (-1 + 1j)/np.sqrt(2)
        constelatons.append('B')
        #print((-1 + 1j)/np.sqrt(2))
    if (x == ['1', '1']).all():
        gray_mapping[index] = (-1 - 1j)/np.sqrt(2)
        constelatons.append('C')
    if (x == ['1', '0']).all():
        gray_mapping[index] = (1 - 1j)/np.sqrt(2)
        constelatons.append('D')

np.savetxt("constellations.txt", constelatons, delimiter="", fmt='%s')
    
"""


print(np.array(gray_mapping))
print(len(gray_mapping))
print(len(data_in_binary))
symbol = np.split(np.array(gray_mapping), len(gray_mapping)/511)


for index, x in enumerate(symbol):
    x = np.reshape(x, 511)
    #x = np.pad(x, (1000, 0), "constant", constant_values=(1, 1))
    conj = np.conjugate(x)[::-1]
    symbol[index] = np.concatenate((x, conj), axis=None)
    symbol[index] = np.insert(symbol[index], 0, 0)
    symbol[index] = np.insert(symbol[index], 512, 0)

print(len(symbol[0]))
print(symbol[0])
info = np.fft.ifft(symbol) # should I iDFT the whole block or iDFT each 1024 symbol, is there a difference?
output = open("info.txt", "w")
df = pd.DataFrame(info[0])
df.to_csv("info.csv", header=False, index=False)

print(info.shape)
to_transmit = np.zeros(shape=(len(info), 1056))

for index, x in enumerate(info):
    cyclic_prefix = x[-32:]
    new_x = np.concatenate((cyclic_prefix, x), axis = None)
    to_transmit[index] = new_x


to_transmit = np.concatenate(to_transmit, axis=0)
print("max")
print(np.max(np.abs(to_transmit)))
print(to_transmit.shape)

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

duration = 1.0  # in seconds, may be float
t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)
samples = scipy.signal.chirp(t, f0=0, f1=5000, t1=int(duration), method='linear').astype(np.float32)

print(samples.shape)

print(to_transmit.shape)
print(to_transmit)
to_transmit = to_transmit.astype(np.float32)
to_transmit = to_transmit / np.max(np.abs(to_transmit))
sound_to_send = np.concatenate((samples, to_transmit))

print(sound_to_send.shape)

to_transmit = to_transmit.astype(np.float32)
sound_to_send = sound_to_send.astype(np.float32)
# generate_sound(sound_to_send, 1, fs)

#np.savetxt("foo.csv", to_transmit, delimiter="")

#generate_sound(sound_to_send, 1, fs)
sound_to_send = sound_to_send.astype(np.float32)
np.savetxt("foo.csv", to_transmit, delimiter="")

# generate_sound(sound_to_send, 1, fs)
"""