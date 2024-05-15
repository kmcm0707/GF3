import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import scipy

def cross_power_spectrum_channel_estimation(x, y):
    # x and y are the time signals to be compared
    # fs is the sampling frequency
    # nfft is the number of points to use in the fft
    #nspre is the number of points to overlap in the fft
    cross_power = signal.csd(x, y, fs=44100, nfft=441000, return_onesided=False)
    x_power = signal.welch(x, fs=44100, nfft=441000, return_onesided=False)

    return cross_power / x_power

def standered_estimation(x, y):
    # x and y are the time signals to be compared
    # needs to be edited if x and y are multblocks
    X = np.fft.fft(x, n=441000)
    Y = np.fft.fft(y, n=441000)
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

    y = pd.read_csv('Initial_Tests/chrip_1-16k_time_data.csv', header=None).to_numpy()
    y= np.reshape(y, len(y))
    y = y[:len(y)//2]
    y= y[2000:]
    duration = 10.0  # in seconds, may be float
    fs = 44100  # sampling rate, Hz, must be integer
    t = np.linspace(0, int(duration), int(fs*duration), endpoint=False)
    x = scipy.signal.chirp(t, f0=1000, f1=16000, t1=int(duration), method='linear').astype(np.float32)
    y = np.pad(y, (0, len(x) - len(y)))
    print(len(x))
    plt.plot(y)
    plt.show()

    chanel_estimation = standered_estimation(x, y)
    print(chanel_estimation)
    #invese_fft = np.fft.fft(y)
    plt.plot(chanel_estimation)
    plt.show()
    