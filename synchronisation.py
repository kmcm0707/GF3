import numpy as np
import scipy.signal
import scipy

def matched_filter_synchronisation(y, duration, sampling_frequency):
    # x and y are the time signals to be compared
    # fs is the sampling frequency
    # needs to be edited if x and y are multblocks
    t = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
    x = scipy.signal.chirp(t, f0=1000, f1=16000, t1=int(duration), method='linear').astype(np.float32)
    y= np.reshape(y, len(y))
    time_for_block = sampling_frequency * duration
    cross_correlation = []
    for i in range(0, len(y) - time_for_block):
        cross_correlation.append(scipy.signal.correlate(y, x, mode='full'))
    cross_correlation = np.array(cross_correlation)
    max_index = np.argmax(cross_correlation)
    return max_index, cross_correlation

