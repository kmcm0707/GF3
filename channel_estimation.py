import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal

def cross_power_spectrum_channel_estimation(x, y):
    # x and y are the time signals to be compared
    # fs is the sampling frequency
    # nfft is the number of points to use in the fft
    #nspre is the number of points to overlap in the fft
    cross_power = signal.csd(x, y, fs=1, nfft=1024, return_onesided=False)
    x_power = signal.welch(x, fs=1, nfft=1024, return_onesided=False)

    return cross_power / x_power

def standered_estimation(x, y):
    # x and y are the time signals to be compared
    # needs to be edited if x and y are multblocks
    X = np.fft.fft(x, n=1024)
    Y = np.fft.fft(y, n=1024)
    channel_estimation = Y / X
    return channel_estimation

def mmse_channel_estimation(x, y):
    # x and y are the time signals to be compared
    # fs is the sampling frequency
    # needs to be edited if x and y are multblocks
    X = np.fft.fft(x, n=1024)
    Y = np.fft.fft(y, n=1024)
    channel_estimation = np.conj(X) @ Y.T @ np.linalg.inv(np.conj(X) @ X.T)
    return channel_estimation

def wiener_hopf_channel_estimation(x, y):
    # needs to be implemented
    return

if __name__ == "__main__":
    # Load file1.csv and channel.csv as vectors

    x = pd.read_csv('inital.csv', header=None).to_numpy()