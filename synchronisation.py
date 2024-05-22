import numpy as np
import scipy.signal
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from channel_estimation_chirp  import *
from scipy.ndimage import uniform_filter1d

def matched_filter_synchronisation(y,x, duration, sampling_frequency):
    # x and y are the time signals to be compared
    # fs is the sampling frequency
    # needs to be edited if x and y are multblocks
    x = np.reshape(x, len(x))
    y= np.reshape(y, len(y))
    cross_correlation = []
    n = len(x)
    N = len(y)
    lags = np.arange(-n + 1, N) 
    cross_correlation.append(scipy.signal.correlate(y, x,mode='full', method='fft'))
        #print(cross_correlation[i])
    cross_correlation = np.array(cross_correlation)
    cross_correlation = uniform_filter1d(cross_correlation, size=5)
    max_index = np.argmax(cross_correlation)
    return max_index, cross_correlation, lags


def matched_filter_synchronisation_differentiation(y,x):
    x = np.reshape(x, len(x))
    y= np.reshape(y, len(y))
    cross_correlation = []
    n = len(x)
    N = len(y)
    lags = np.arange(-n + 1, N)
    cross_correlation.append(scipy.signal.correlate(y, x,mode='full', method='fft'))
    cross_correlation = np.array(cross_correlation)
    cross_correlation = np.reshape(cross_correlation, cross_correlation.shape[1])
    for i in range(len(cross_correlation)):
        if np.abs(cross_correlation[i]) < 0.5:
            cross_correlation[i] = 0
    cross_correlation = np.diff(cross_correlation)
    cross_correlation = cross_correlation
    
    zero_crossings = np.where(np.diff(np.sign(cross_correlation)))[0]
    max_index = zero_crossings[0]
    return max_index, cross_correlation, lags


if __name__ == "__main__":
    y = pd.read_csv('Initial_Tests/chrip_1-16k_time_data.csv', header=None).to_numpy()
    y= np.reshape(y, len(y))
    y = y[:len(y)//2]
    
    
    duration = 10.0  # in seconds, may be float
    fs = 44100  # sampling rate, Hz, must be integer
    t = np.linspace(0, int(duration), int(fs*duration), endpoint=True)

    x = scipy.signal.chirp(t, f0=1000, f1=16000, t1=int(duration), method='linear').astype(np.float32)

    #x = np.sin(2 * np.pi * np.arange(fs * duration) * f / fs).astype(np.float32)
    x = scipy.signal.chirp(t, f0=1000, f1=16000, t1=int(duration), method='linear').astype(np.float32)
    max_index, cross_correlation, lags = matched_filter_synchronisation(y,x, duration, fs)
    
    cross_correlation = np.reshape(cross_correlation, cross_correlation.shape[1])
    print(max_index)
    #max_index, cross_correlation_diff, lags = matched_filter_synchronisation_differentiation(y,x)
    print(max_index)
    plt.plot(cross_correlation)
    plt.vlines(max_index, -cross_correlation[max_index], cross_correlation[max_index], colors='r')
    plt.show()
    correct_data = y[lags[max_index]:lags[max_index] + len(x)]
    print(len(x))
    print(len(correct_data))
    plt.plot(correct_data)
    plt.show()


    channel_estimation = standered_estimation(x, correct_data, 44100)
    
    frequencies = np.fft.fftfreq(n=44100, d=1/44100)
    frequencies = np.fft.fftshift(frequencies)
    plt.plot(np.abs(channel_estimation))
    plt.show()
    plt.plot(np.fft.ifft(channel_estimation))
    plt.show()



