from synchronisation import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from channel_estimation_chirp  import *
from scipy.ndimage import uniform_filter1d

if __name__ == "__main__":
    correct_data = pd.read_csv('bee_data.txt', header=None).to_numpy()
    correct_data = np.reshape(correct_data, len(correct_data))


    y = pd.read_csv('bee_data_2.csv', header=None).to_numpy()
    y = np.reshape(y, len(y))
    plt.plot(y)
    plt.show()


    duration_chirp = 1
    fs = 48000
    t = np.linspace(0, int(duration_chirp), int(fs*duration_chirp), endpoint=True)
    chirp = scipy.signal.chirp(t, f0=0, f1=5000, t1=int(duration_chirp), method='linear').astype(np.float32)

    max_index, cross_correlation, lags = matched_filter_synchronisation(y,chirp, duration_chirp, fs)
    cross_correlation = np.reshape(cross_correlation, cross_correlation.shape[1])
    print(lags[max_index])
    print(len(y))
    print(len(chirp))

 

    correct_data_chirp = y[lags[max_index]:lags[max_index] + len(chirp)]
    plt.plot(correct_data_chirp)
    plt.show()
    chanel_est = standered_estimation(chirp, correct_data_chirp, fs)

    bee_len = 1152112 
    signal_data = y[lags[max_index]+len(chirp):lags[max_index] +len(chirp) + bee_len]
    signal_data = np.pad(signal_data, (0, 1056 - len(signal_data) % 1056), 'constant')

    signal_data = np.reshape(signal_data, len(signal_data))
    signal_data = signal_data 
    plt.plot(signal_data)
    plt.show()

    


    symbol_len = 1056
    no_symbols = int(len(signal_data) / symbol_len)
    #signal_data = signal_data.pad(signal_data, (0, 1024 - len(signal_data)), "constant")
    symbols = np.split(signal_data, len(signal_data)/symbol_len)


    for index, i in enumerate(symbols):
        symbols[index] = i[32:]
    
    symbols_freq = np.ones((no_symbols, 1024))
    symbols_freq = symbols_freq.astype(complex)
    for index, i in enumerate(symbols):
        temp = np.reshape(i, 1024)
        symbols_freq[index] = np.fft.fft(temp)

    plt.plot(np.abs(chanel_est))
    plt.show()
    chanel_time = np.fft.ifft(chanel_est)
    plt.plot(chanel_time)
    plt.show()
    chanel_time = chanel_time[:3024]
    #chanel_time = np.reshape(chanel_time, 1024)
    chanel_freq = np.fft.fft(chanel_time)
    plt.plot(np.abs(chanel_freq))
    plt.show()
    plt.plot(np.abs(symbols_freq[0]))
    plt.show()
    recieved_freq = symbols_freq / chanel_freq
    constellations = np.zeros((1, 511))
    for index, i in enumerate(recieved_freq):
        constellations = np.vstack((constellations,i[1:512]))

    constellations = np.delete(constellations, 0, 0)
    print(constellations)
    #for i in constellations:
    index = 0
    for i in range(0,10):
        for i in constellations[i]:
            if correct_data[index] == 0 and correct_data[index+1] == 0:
                plt.scatter(i.real, i.imag, color = 'red')
            elif correct_data[index] == 0 and correct_data[index+1] == 1:
                plt.scatter(i.real, i.imag, color = 'blue')
            elif correct_data[index] == 1 and correct_data[index+1] == 1:
                plt.scatter(i.real, i.imag, color = 'green')
            elif correct_data[index] == 1 and correct_data[index+1] == 0:
                plt.scatter(i.real, i.imag, color = 'yellow')
            index += 2

        plt.show()
    
    plt.show()

    binary = []

    for symbol in constellations:
        for i in symbol:
            if np.real(i) >= 0 and np.imag(i) >= 0:
                binary.append("00")
            elif np.real(i) <= 0 and np.imag(i) >= 0:
                binary.append("01")
            elif np.real(i) <= 0 and np.imag(i) <= 0:
                binary.append("11")
            elif np.real(i) >= 0 and np.imag(i) <= 0:
                binary.append("10")
            else:
                print("uh oh!")
        
    binary = ''.join(binary)
    # output = binary.decode("ascii")

    output = open("output_bee.txt", "w")
    output.write(binary)
    output.close()



    

