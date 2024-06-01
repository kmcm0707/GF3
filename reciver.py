import sounddevice as sd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

def reciver(fs, time, save_file = False, file_name = 'recording.wav'):
    numsamples = int(fs * time)
    print("recording")
    recording = sd.rec(numsamples, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("done recording")
    data = np.array(recording)
    data = data.flatten()
    if save_file:
        pd.DataFrame(data).to_csv(file_name, index = False, header = False)
    return data

if __name__ == "__main__":
    fs = 48000
    time = 15 # seconds
    recording = reciver(fs, time, save_file = True, file_name = 'recording_2.csv')
    print(recording)
    plt.plot(recording)
    plt.show()