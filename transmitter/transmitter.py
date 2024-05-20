import numpy as np

import time
import pyaudio
import scipy

# from sound_generator import *


with open('transmitter/data_file.txt', 'r') as file:
    data = file.read().replace('\n', '')

print(data)
print(type(bytearray(data, 'utf-8')))

data_in_binary = (''.join(format(x, 'b') for x in bytearray(data, 'utf-8')))
data_in_binary += ''.join(str(x) for x in ([0] * (1022 - len(data_in_binary) % 1022)))
data_in_binary = np.array(list(data_in_binary))
print(data_in_binary)
print(len(data_in_binary))
gray_mapping = np.split(data_in_binary, len(data_in_binary)/2)
for index, x in enumerate(gray_mapping):
    if tuple(x) == ('0','0'):
        gray_mapping[index] = (1 + 1j)/np.sqrt(2)
    if tuple(x) == ('0','1'):
        gray_mapping[index] = (-1 + 1j)/np.sqrt(2)
    if tuple(x) == ('1','1'):
        gray_mapping[index] = (-1 - 1j)/np.sqrt(2)
    else:
        gray_mapping[index] = (1 - 1j)/np.sqrt(2)

symbol = np.split(np.array(gray_mapping), len(gray_mapping)/511)

for index, x in enumerate(symbol):
    conj = np.conjugate(x)[::-1]
    symbol[index] = np.concatenate((x, conj), axis=None)
    symbol[index] = np.insert(symbol[index], 0, 0)
    symbol[index] = np.insert(symbol[index], 512, 0)

info = np.fft.ifft(symbol) # should I iDFT the whole block or iDFT each 1024 symbol, is there a difference?

to_transmit = np.zeros(shape=(len(info), 1057))

for index, x in enumerate(info):
    cyclic_prefix = x[-33:]
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

duration = 5.0  # in seconds, may be float
t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)
samples = scipy.signal.chirp(t, f0=1000, f1=16000, t1=int(duration), method='linear').astype(np.float32)

print(samples.shape)

print(to_transmit.shape)

sound_to_send = np.concatenate((samples, to_transmit))

print(sound_to_send.shape)

generate_sound(sound_to_send, 1, fs)