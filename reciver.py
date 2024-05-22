import sounddevice as sd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

def reciver(fs, time, save_file = False, file_name = 'recording.wav'):
    numsamples = int(fs * time)
    recording = sd.rec(numsamples, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    if save_file:
        output = open(file_name, "w")
        output.write(recording)
        output.close()
    return recording

        