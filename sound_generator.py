import time
import numpy as np
import pyaudio
import scipy
import scipy.signal as signal
import pandas as pd

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


duration = 10.0  # in seconds, may be float
f = 4000.0  # sine frequency, Hz, may be float

# generate samples from file1, note conversion to float32 array
"""dataframe1 = pd.read_csv('weekend_files/file1.csv', header=None)
to_decode = dataframe1.to_numpy()
samples = np.reshape(to_decode, len(to_decode)).astype(np.float32)
#samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

# generate chirp signal
t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)"""
#samples = signal.chirp(t, f0=0, f1=16000, t1=int(duration), method='linear').astype(np.float32)
# per @yahweh comment explicitly convert to bytes sequence
duration = 5.0  # in seconds, may be float
fs = 44100  # sampling rate, Hz, must be integer
t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)
#samples = scipy.signal.chirp(t, f0=1000, f1=16000, t1=int(duration), method='linear').astype(np.float32)
sin = np.sin(2 * np.pi * np.arange(fs * duration) * 4000 / fs).astype(np.float32)
generate_sound(sin, 0.5, fs)

