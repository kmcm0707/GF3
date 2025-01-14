import numpy as np
from audio_modem import audio_modem
from scipy import signal
import scipy
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans
from scipy import stats
import math
import matplotlib.pyplot as plt

class receiver(audio_modem):
    def __init__(self):   
        audio_modem.__init__(self)

        self.channel_freq = None
        self.constellations = []
        self.data_index = None
        self.past_centers = [] # Phase Correction Parameters
        self.past_angle = 0
        self.past_gradient = 0
        self.past_change = 0
        self.entire_data = None
        self.bits = None # Header Information
        self.file_name = None
        self.sigma2 = None
        # self.pre_ldpc_data = []
        self.corrected = []
        self.first_decoded = []

        self.iterative_ldpc_weight = 0.6

        # Generate Watermark
        self.watermark = self.known_ofdm_block_mod4

    def set_bits_and_file_name(self, bits, file_name):
        self.bits = bits
        self.file_name = file_name
    
    def find_start_index(self, data, position="start"):
        """ Find Index of Chirp Start. Returns index of data where chirp starts, and the cross_correlation graph """
        alldata = data

        if position == "start":
            data = data[:200000] # Ensure only capture the first chirp
        elif position == "end":
            data = data[200000:] # Ensure only capture the last chirp

        cross_correlation = scipy.signal.correlate(data, self.chirp ,mode='full', method='fft')
        cross_correlation = np.array(cross_correlation)

        cross_correlation = np.abs(cross_correlation)
        cross_correlation = uniform_filter1d(cross_correlation, size=5)

        max_index = np.argmax(cross_correlation)

        n = len(self.chirp)
        N = len(data)

        if position == "start":
            chirp_start = np.arange(-n + 1, N)[max_index]
        elif position == "end":
            chirp_start = len(alldata) - N + np.arange(-n + 1, N)[max_index]

        # Plot the data with a colour coded chirp
        positions = np.arange(0, len(alldata))
        plt.plot(positions[:chirp_start - 1024], alldata[:chirp_start - 1024])
        plt.plot(positions[chirp_start - 1024 : chirp_start + len(self.chirp_p_s) - 1024], alldata[chirp_start - 1024 : chirp_start+len(self.chirp_p_s) - 1024], color='r')
        plt.plot(positions[chirp_start + len(self.chirp_p_s) - 1024:], alldata[chirp_start + len(self.chirp_p_s) - 1024:], color='g')
        plt.show()

        return chirp_start, cross_correlation
    
    def channel_estimation(self, block, ideal_block, mode="basic"):
        """ Find Channel Response - find from received OFDM block and expected (tranmsitted) OFDM block in freq domain """

        # Y = Recieved OFDM Blocks  "block"
        # X = Ideal OFDM Blocks     "ideal_block"
        # H = Channel Response

        if mode == "basic":
            # H = Y/X
            channel_freq = np.true_divide(block, ideal_block, out=np.ones_like(block), where=ideal_block!=0).astype(complex)
        elif mode == "wiener":
            lambda_val = 1000
            channel_freq = np.true_divide(block * np.conj(ideal_block), (ideal_block * np.conj(ideal_block) * lambda_val), where=ideal_block!=0, out=np.ones_like(block)).astype(complex)
        else:
            raise Exception("Only basic and wiener are valid channel_estimation modes!")

        channel_freq[0] = 1
        channel_freq[self.ofdm_symbol_size // 2] = 1

        self.channel_freq = channel_freq

        return channel_freq

    def find_data_index(self, data, start_index):
        """ Find Data Start """
        self.data_index = start_index + len(self.chirp_p_s) - self.ofdm_prefix_size
        return self.data_index
    
    def calculate_sigma2_five_block(self, recieved, ideal):
        """generate sigma2 with five blocks - ideal = [5, 4096] (e.g. all the same) (generated block - all freq bins); recieved = [5, 648]"""

        ideal = ideal * self.channel_freq
        recieved = np.array(recieved)
        ideal = np.array(ideal)

        print("Ideal", ideal)
        print("recieved", recieved)

        ideal = ideal[:, self.ofdm_bin_min:self.ofdm_bin_max+1]
        real_square_error = (recieved.real - ideal.real) ** 2
        real_square_error = real_square_error.astype(np.float32)

        imag_square_error = (recieved.imag - ideal.imag) ** 2
        imag_square_error = imag_square_error.astype(np.float32)

        all_errors = np.concatenate((real_square_error, imag_square_error))

        sigma2 = np.mean(all_errors)
        print("second guess sigma2", sigma2) # Should be ~ 0.1 ish for ideal channel?

        return sigma2

    def combined_correction(self, current_OFDM): # TODO Cleanup?
        past_angle = self.past_angle
        past_gradient = self.past_gradient
        past_centers = self.past_centers
        past_change = self.past_change

        corrected = current_OFDM
        corrected = corrected * np.exp(-1j * past_gradient)
        corrected = corrected * np.exp(-1j * past_angle)
        centers = []
        d = []
        past_centers = np.asarray(past_centers)
        labels = []
        inti = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        inti = np.array(inti)
        inti_complex = inti[:, 0] + 1j * inti[:, 1]
        if past_centers.shape[0] != 0:
            kmeans = KMeans(n_clusters=4, init=inti, random_state=0 ).fit(np.array([np.real(corrected), np.imag(corrected)]).T)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            complex_centers = centers[:, 0] + 1j * centers[:, 1]
            complex_angles = np.angle(complex_centers)
            inti_angles = np.angle(inti_complex)
            angle_diff = complex_angles - inti_angles
            angle = np.array(angle_diff)
            angle = np.mean(angle) #- np.pi
            corrected = corrected * np.exp(-1j * angle)
            past_angle += angle 
        else:
            kmeans = KMeans(n_clusters=4, init=inti ).fit(np.array([np.real(corrected), np.imag(corrected)]).T)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            complex_centers = centers[:, 0] + 1j * centers[:, 1]

        predicted_ideal_angle = []
        for i in range(len(labels)):
            predicted_ideal_angle.append(np.angle(complex_centers[labels[i]]))
        predicted_ideal_angle = np.array(predicted_ideal_angle)
        predicted_ideal_angle = np.unwrap(predicted_ideal_angle)
        known_angles = np.angle(corrected)
        known_angles = np.unwrap(known_angles)
        diff = known_angles - predicted_ideal_angle
        diff = np.unwrap(diff)


        for index, i in enumerate(diff):
            if i > np.pi + 1:
                diff[index] = diff[index] - 2 * np.pi
            if i < -np.pi - 1:
                diff[index] = diff[index] + 2 * np.pi
        positions = np.arange(len(corrected)) 
        grad, intercept, r_value, p_value, std_err = stats.linregress(positions, diff)
        if math.isclose(intercept, 2* np.pi, abs_tol=2):
            intercept = intercept - 2 * np.pi
        if math.isclose(intercept, -2* np.pi, abs_tol=2):
            intercept = intercept + 2 * np.pi

        gradient_2 = grad * np.arange(len(current_OFDM)) #+ intercept

        if np.mean(np.abs(gradient_2)) > 1:
            gradient_2 = self.past_change
        corrected = corrected * np.exp(-1j * gradient_2)
        gradient = gradient_2 + past_gradient

        self.past_change = gradient_2

        self.past_centers = centers
        self.past_angle = past_angle
        self.past_gradient = gradient
        return corrected

    def ofdm_one_block(self, data_block, sigma2):
        """Decode one block of data - returns decoded bits (with parity) and LLRs"""
        assert len(data_block) == self.ofdm_symbol_size

        # DFT
        freq = np.fft.fft(data_block)
        
        # Divide by Channel Response
        freq = freq / self.channel_freq

        # Remove Complex Conjugate Bins
        freq = freq[1:2048]

        # Rotate by Watermark
        freq = freq * np.exp(-1j * self.watermark * np.pi / 2)

        # Phase correction
        corrected = self.combined_correction(freq[self.ofdm_bin_min-1:self.ofdm_bin_max])
        self.corrected.append(corrected * self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1])

        # Find Closest Constellation Point and LLRs
        decoded = []
        llrs = []
        for index, i in enumerate(corrected):
            decoded.extend(self.constellation_point_to_binary(i))

            # If a point is really far away, it is probably wrong (limits really great LLRs)
            imag = np.imag(i)
            if np.abs(np.imag(i)) > 5:
                imag = 5 * np.sign(imag)
            real = np.real(i)
            if np.abs(np.real(i)) > 5:
                real = 5 * np.sign(real)

            # Find LLRs by using distance from axes for soft LDPC decoding:
            # L_1(y) = c_k \times c_k^* y'_i / \sigma ^ 2
            l1 = self.channel_freq[index + self.ofdm_bin_min] * np.conj(self.channel_freq[index + self.ofdm_bin_min]) * imag / sigma2
            # L_2(y) = c_k \times c_k^* y'_r / \sigma ^ 2 ... from Jossy LDPC Paper
            l2 = self.channel_freq[index + self.ofdm_bin_min] * np.conj(self.channel_freq[index + self.ofdm_bin_min]) * real / sigma2

            llrs.extend([l1.real, l2.real])

        return decoded, llrs

    def ldpc_decode_one_block(self, to_decode, llrs, mode="soft"):
        """ Decode one block of data (LDPC), returns decoded bits"""
        to_decode = np.array(to_decode)
        to_decode = np.reshape(to_decode, len(to_decode))
        llrs = np.array(llrs)
        llrs = np.reshape(llrs, len(llrs))
        decoded = []

        assert len(to_decode) == len(llrs)
        assert len(to_decode) == self.c.N

        if mode == "soft":
            decoded_block, iters = self.c.decode(llrs)
            decoded_temp = []
            decoded_temp += ([1 if k < 0 else 0 for k in decoded_block])
            self.first_decoded.append(decoded_temp)

            decoded_block = decoded_block[:-(self.c.K)] # No idea what the extra information is
            decoded += ([1 if k < 0 else 0 for k in decoded_block])

        elif mode == "hard":
            i = to_decode.copy()
            i = 10 * (0.5 - i) # Do weightings

            decoded_block, iters = self.c.decode(i)
            decoded_block = decoded_block[:-(self.c.K)] # No idea what the extra information is
            decoded += ([1 if k < 0 else 0 for k in decoded_block])
        else:
            raise Exception("Only 'hard' and 'soft' are valid ldpc decoding modes")
            
        return decoded

    def data_block_processing(self, data_block, known_blocks = 5):
        """" Processes the block of data after synchronisation """
        all_data = []

        ofdm_symbols = [] # List of all the ofdm_symbols

        # Find OFDM symbols

        # data_block = data_block[:10*(self.ofdm_symbol_size + self.ofdm_prefix_size)] # TODO REMOVE!!! FOR TESTING

        print("Len of Data_block:", len(data_block))

        assert len(data_block) % (self.ofdm_symbol_size + self.ofdm_prefix_size) == 0
        number_ofdm_symbols = len(data_block) // (self.ofdm_symbol_size + self.ofdm_prefix_size)

        print("Number of OFDM Symbols:", number_ofdm_symbols)

        for i in range(number_ofdm_symbols):
            start_index = i * (self.ofdm_symbol_size + self.ofdm_prefix_size) + self.ofdm_prefix_size
            end_index = (i+1) * (self.ofdm_symbol_size + self.ofdm_prefix_size)
            ofdm_symbols.append(data_block[start_index : end_index])

        assert len(ofdm_symbols[0]) == self.ofdm_symbol_size

        # Initial channel estimate using 1st known OFDM block:
        channel_freq = self.channel_estimation(np.fft.fft(ofdm_symbols[0]), self.known_ofdm_block)
        self.channel_freq = channel_freq
        print(self.channel_freq)

        # Do initial phase correction for first ofdm block:
        if known_blocks == 1:
            ofdm_freq = np.fft.fft(ofdm_symbols[0])
            self.sigma2 = 10 # Initial sigma guess (for 1 block) works :)
            print("Initial sigma2 estimate = ", self.sigma2)
            ofdm_freq = ofdm_freq / channel_freq
            corrected = self.combined_correction(ofdm_freq[self.ofdm_bin_min : self.ofdm_bin_max + 1])

        elif known_blocks == 5:
            channel_freqs = []
            ofdm_freqs = []

            for i in ofdm_symbols[0:5]:
                ofdm_freq = np.fft.fft(i)
                ofdm_freqs.append(ofdm_freq)
                channel_freqs.append(self.channel_estimation(ofdm_freq, self.known_ofdm_block))

            channel_freqs = np.array(channel_freqs)

            # print(channel_freqs, channel_freqs.shape)
            channel_freq = np.mean(channel_freqs, axis=0)
            self.channel_freq = channel_freq

            assert channel_freq.shape[0] == 4096

            ofdm_freqs = ofdm_freqs / channel_freq
            ofdm_freqs = ofdm_freqs[:, 1:2048]

            corrected_symbols = []
            for i in ofdm_freqs:
                corrected_symbols.append(self.combined_correction(i[self.ofdm_bin_min - 1: self.ofdm_bin_max]))

            ideal_block = self.known_ofdm_block 
            
            self.sigma2 = self.calculate_sigma2_five_block((ofdm_freqs * channel_freq[1:2048])[:,self.ofdm_bin_min - 1: self.ofdm_bin_max], [ideal_block, ideal_block, ideal_block, ideal_block, ideal_block])

            # self.sigma2 = 10

        else:
            raise Exception("Values of known blocks apart from 1 or 5 not supported")

        # Decode the OFDM Symbols:

        for index, symbol in enumerate(ofdm_symbols[known_blocks:]): # TODO change for 5 blocks
            assert len(symbol) == self.ofdm_symbol_size

            print("Decoding symbol no. ", index)

            last_decoded = np.zeros_like(self.all_bins * 2)

            for i in range(50): # limit max iterations
                # Decode Block
                data_bins, llrs = self.ofdm_one_block(symbol, self.sigma2)
                # self.pre_ldpc_data.extend(data_bins) # For Testing
                decoded = self.ldpc_decode_one_block(data_bins, llrs)

                constellation = self.binary_to_constellation_point(self.first_decoded[-1])

                # Iterative LDPC: update channel coefficients
                self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1] = self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1] * self.iterative_ldpc_weight + (1-self.iterative_ldpc_weight) * self.corrected[-1] / constellation

                decoded = np.array(decoded)
                last_decoded = np.array(last_decoded)

                if (last_decoded == decoded).all(): # Detect if iterative LDPC changed anything?
                    break
                
                if i == 49:
                    print("Broken because of max iterations")

                last_decoded = decoded 
            
            self.constellations.extend(self.corrected[-1] / self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1])
            all_data.extend(last_decoded)
        
        self.corrected = np.array(self.corrected)

        all_data = all_data[:self.bits]
        return all_data
     
    def extract_header(self, data):
        """ Takes data as a list of bits, and returns the list of bits without the header. Sets header info to object data """
        # Extract Header
        data = np.array(data)
        data = np.reshape(data, len(data))

        null_character = np.zeros(8)

        # Iterate through start of data finding header:
        num_nulls = 0
        name_start = 0
        header_start = 0
        name = []
        header = []
        restofdata = []

        for i in range(0, len(data), 8):
            if ((data[i:i+8] == null_character).all()):
                num_nulls += 1
                if num_nulls == 2:
                    name_start = i+8
                if num_nulls == 3:
                    name.extend(data[name_start:i])
                if num_nulls == 4:
                    header_start = i+8
                if num_nulls == 5:
                    header.extend(data[header_start:i])
                if num_nulls == 6:
                    restofdata.extend(data[i+8:])
                    break

        file_name = self.decode_text(name)
        bits = self.decode_text(header)

        print("File Name: ", file_name)
        print("Header: ", bits)

        self.set_bits_and_file_name(bits, file_name)
        return restofdata
        
    def decode_text(self, binary_data):
        """ decodes binary data to string through ascii encoding """
        binary_data = np.array(binary_data).astype("str")
        ascii = [int(''.join(binary_data[i:i+8]), 2) for i in range(0, len(binary_data), 8)]
        return ''.join([chr(i) for i in ascii])
    
    def save_decoded_file(self, data_arr, size):
        """ saves (binary) data to self.file_name """
        data = ''.join(str(i) for i in data_arr)
        byte = int(data, 2).to_bytes(len(data) // 8, byteorder='big')
        output = open("outputs/" + self.file_name, "wb")
        output.write(byte)
        output.close()

    def read_wav(self, filename):
        samplerate, data = scipy.io.wavfile.read(filename)
        print("Sample rate (wav) = ", samplerate)
        return data

def success(a, b):
    """find the percentage difference between two lists"""
    successes = 0
    wrong_indices = []
    for index, i in enumerate(a):
        if i == b[index]:
            successes += 1 / len(a)
        else:
            wrong_indices.append(index)
    return successes

from transmitter import transmitter

if __name__ == "__main__":
    r = receiver()

    entire_data = np.loadtxt('data/benson_testing1.csv', delimiter = ",", dtype = "float")
    # entire_data = r.read_wav("data/recording_64.wav")

    ### FIND CROSS CORRELATIONS (start and end chirps)

    start_index, cross_correlation = r.find_start_index(entire_data)

    # plt.plot(cross_correlation)
    # plt.title("Cross correlation for start chirp")
    # plt.show()

    end_index, cross_correlation = r.find_start_index(entire_data, position = "end")

    ### FIND LENGTH AND SEPERATE DATA BLOCK

    end_index = end_index - r.ofdm_prefix_size # ???
    data_index = r.find_data_index(entire_data, start_index)
    data_length = end_index - data_index

    print("Data Length:", data_length)

    num_symbols = data_length // (r.ofdm_symbol_size + r.ofdm_prefix_size) # Floor division may mean that we loose some data at the end?
    data_block = entire_data[data_index : data_index + (num_symbols) * (r.ofdm_symbol_size + r.ofdm_prefix_size)]

    r.bits = int(num_symbols * r.c.K)
    print("Bits:", r.bits)

    data = r.data_block_processing(data_block)

    print("Data: ", data[0:100])
    print(r.decode_text(data)[0:100])

    data_without_header = r.extract_header(data)
    r.save_decoded_file(data_without_header, r.bits)

    t =  transmitter()

    transmitted_bits = t.process_file("data/hamlet_in.txt")
    binary_data_with_header = t.add_header(transmitted_bits)
    ldpc_bits = t.ldpc_encode(binary_data_with_header) # Data from transmitter before LDPC encoding

    # ldpc_bits_r = r.pre_ldpc_data # Data from receiver before LDPC decoding

    colors = []

    for index in range(0, len(ldpc_bits), 2):
        if (ldpc_bits[index], ldpc_bits[index+1]) == (0, 0):
            colors.append('r')
        elif (ldpc_bits[index], ldpc_bits[index+1]) == (0, 1):
            colors.append('g')
        elif (ldpc_bits[index], ldpc_bits[index+1]) == (1, 1):
            colors.append('b')
        elif (ldpc_bits[index], ldpc_bits[index+1]) == (1, 0):
            colors.append('y')

    for index in [0, 10, 50]: # Iterate through the OFDM Symbols
        plt.scatter(np.real(r.constellations[648 * index:648 * (index+1)]), np.imag(r.constellations[648 * index:648 * (index+1)]), c=colors[648 * index:648 * (index+1)])
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.title("OFDM block number - " + str(index))

        ax = plt.gca() # Limit Axes size so don't have to keep zooming in
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])

        plt.show()

        print("---- OFDM Block Number ---- = ", index)
        # print("Pre-LDPC Success", success(ldpc_bits[index*648*2:(index+1)*648*2], ldpc_bits_r[index*648*2:(index+1)*648*2]))
        print("Post-LDPC Success", success(binary_data_with_header[index*648:(index+1)*648], data[index*648:(index+1)*648]))


