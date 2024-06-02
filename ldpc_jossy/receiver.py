import numpy as np
from audio_modem import audio_modem
from scipy import signal
import scipy
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans
from scipy import stats
import math

class receiver(audio_modem):
    def __init__(self):
        
        audio_modem.__init__(self)
        self.channel_freq = None
        self.constellations = None
        self.data_index = None
        self.chirp_start = None
        self.past_centers = []
        self.past_angle = 0
        self.past_gradient = 0
        self.entire_data = None
        self.bits = None
        self.file_name = None

    def set_bits_and_file_name(self, bits, file_name):
        self.bits = bits
        self.file_name = file_name
    
    def find_start_index(self, data):
        # Find Chirp Start
        # x and y are the time signals to be compared
        x = self.chirp
        cross_correlation = []
        n = len(x)
        N = len(data)
        lags = np.arange(-n + 1, N) 
        cross_correlation.append(scipy.signal.correlate(data, x,mode='full', method='fft'))
        cross_correlation = np.array(cross_correlation)
        cross_correlation = uniform_filter1d(cross_correlation, size=5)
        max_index = np.argmax(cross_correlation)
        self.chirp_start = max_index
        return max_index, cross_correlation, lags
    
    def channel_estimation(self, block, ideal_block):
        # Find Channel Response
        # H = Y/X
        # Y = Recieved OFDM Blocks
        # X = Ideal OFDM Blocks
        # H = Channel Response
        self.channel_freq = np.true_divide(block, ideal_block, out=np.zeros_like(block), where=ideal_block!=0).astype(complex)
        self.channel_freq[0] = self.channel_freq[1]
        self.channel_freq[self.ofdm_symbol_size // 2] = self.channel_freq[self.ofdm_symbol_size // 2 - 1]
        return self.channel_freq

    def find_data_index(self, data, start_index):
        # Find Data Start
        self.data_index = start_index + len(self.chirp_p_s)
        return self.data_index
    
    def calculate_sigma2(self, recieved, ideal):
        # TODO: CHECK WITH MAX CORRECT
        # Calculate Noise Power
        # sigma2 = 1/N * \sum_{k=1}^{N} |Y_k - H_kX_k|^2
        # Y = Recieved OFDM Blocks
        # X = Ideal OFDM Blocks
        # H = Channel Response
        # sigma2 = Noise Power
        ideal = ideal * self.channel_freq[self.ofdm_bin_min - 1:self.ofdm_bin_max]
        sigma2 = np.mean(np.abs(recieved - ideal) ** 2) 
        
        print("first guess sigma2:", sigma2)

        print(recieved[0:10])
        print("---")
        print(ideal[0:10])

        real_square_error = (recieved.real - ideal.real) ** 2
        real_square_error = real_square_error.astype(np.float32)
        # print(real_square_error)
        imag_square_error = (recieved.imag - ideal.imag) ** 2
        imag_square_error = imag_square_error.astype(np.float32)
        # print(imag_square_error)
        all_errors = np.concatenate((real_square_error, imag_square_error))

        sigma2 = np.mean(all_errors)
        print("second guess sigma2", sigma2) # Should be ~ 0.1 ish for ideal channel?
        return sigma2
    
    def calculate_sigma2_one_block(self, recieved):
        dec = []
        for i in recieved:
            if np.real(i) >= 0 and np.imag(i) >= 0:
                dec.append(1 + 1j)
            elif np.real(i) <= 0 and np.imag(i) >= 0:
                dec.append(-1 + 1j)
            elif np.real(i) <= 0 and np.imag(i) <= 0:
                dec.append(-1 - 1j)
            elif np.real(i) >= 0 and np.imag(i) <= 0:
                dec.append(1 - 1j)
        dec = np.array(dec)

        real_square_error = (recieved.real - dec.real) ** 2
        real_square_error = real_square_error.astype(np.float32)
        # print(real_square_error)
        imag_square_error = (recieved.imag - dec.imag) ** 2
        imag_square_error = imag_square_error.astype(np.float32)

        all_errors = np.concatenate((real_square_error, imag_square_error))

        sigma2 = np.mean(all_errors)
        return sigma2
            
    
    def combined_correction(self, current_OFDM):
        past_angle = self.past_angle
        past_gradient = self.past_gradient
        past_centers = self.past_centers

        corrected = current_OFDM
        corrected = corrected * np.exp(-1j * past_gradient)
        corrected = corrected * np.exp(-1j * past_angle)
        centers = []
        d = []
        past_centers = np.asarray(past_centers)
        """cluster_1 = []
        cluster_2 = []
        cluster_3 = []
        cluster_4 = []"""
        labels = []
        inti = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        inti = np.array(inti)
        inti_complex = inti[:, 0] + 1j * inti[:, 1]
        if past_centers.shape[0] != 0:
            kmeans = KMeans(n_clusters=4, init=inti, random_state=0 ).fit(np.array([np.real(corrected), np.imag(corrected)]).T)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            complex_centers = centers[:, 0] + 1j * centers[:, 1]
            distance = complex_centers - inti_complex
            d = np.array(distance)
            angle = np.angle(d)
            for i in range(len(angle)):
                if angle[i] < 0:
                    angle[i] = 2 * np.pi + angle[i]
            angle = np.mean(angle) - np.pi
            #print(angle)
            corrected = corrected * np.exp(-1j * angle)
            """angle = np.mean(angle)
            if angle < 0:
                angle = 2 * np.pi + angle
            corrected = corrected * np.exp(-1j * angle)"""
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
        positions = np.arange(len(corrected)) + 1
        grad, intercept, r_value, p_value, std_err = stats.linregress(positions, diff)
        if math.isclose(intercept, 2* np.pi, abs_tol=2):
            intercept = intercept - 2 * np.pi
        if math.isclose(intercept, -2* np.pi, abs_tol=2):
            intercept = intercept + 2 * np.pi

        gradient = grad * np.arange(len(current_OFDM)) + intercept

        if np.mean(np.abs(gradient)) > 0.6:
            gradient = 0
        corrected = corrected * np.exp(-1j * gradient)
        gradient = gradient + past_gradient

        self.past_centers = centers
        self.past_angle = past_angle
        self.past_gradient = gradient
        return corrected

    def ofdm_one_block(self, data_block, sigma2):
        """Decode one block of data"""
        assert len(data_block) == self.ofdm_symbol_size

        # DFT
        freq = np.fft.fft(data_block)
        
        # Divide by Channel Response
        freq = freq / self.channel_freq

        # Remove Complex Conjugate Bins
        freq = freq[1:2048]

        corrected = self.combined_correction(freq)

        # Remove Watermark
        corrected = corrected[self.ofdm_bin_min - 1:self.ofdm_bin_max]

        assert len(corrected) == self.bin_length

        watermark = self.generate_known_ofdm_block_mod4()
        watermark = watermark[self.ofdm_bin_min-1:self.ofdm_bin_max]
        print("Watermark: ", watermark[0:5])

        # Rotate Watermark
        corrected = corrected * np.exp(-1j * watermark * np.pi / 2)

        self.constellations = corrected

        # Find Closest Constellation Point
        decoded = []
        llrs = []
        for index, i in enumerate(corrected):
            decoded.extend(self.constellation_point_to_binary(i))

            # Find LLRs by using distance from axes for soft LDPC decoding:
            # L_1(y) = c_k \times c_k^* y'_i / \sigma ^ 2
            l1 = self.channel_freq[index + self.ofdm_bin_min] * np.conj(self.channel_freq[index + self.ofdm_bin_min]) * np.imag(i) / sigma2
            # L_2(y) = c_k \times c_k^* y'_r / \sigma ^ 2 ... from Jossy LDPC Paper
            l2 = self.channel_freq[index + self.ofdm_bin_min] * np.conj(self.channel_freq[index + self.ofdm_bin_min]) * np.real(i) / sigma2
            llrs.extend([l1.real, l2.real])

        return decoded, llrs

    def ldpc_decode_one_block(self, to_decode, llrs, mode="soft"):
        to_decode = np.array(to_decode)
        to_decode = np.reshape(to_decode, len(to_decode))
        llrs = np.array(llrs)
        llrs = np.reshape(llrs, len(llrs))
        decoded = []

        assert len(to_decode) == len(llrs)
        assert len(to_decode) == self.c.N

        if mode == "soft":
            decoded_block, iters = self.c.decode(llrs)

            # print("Iterations ", iters)

            decoded_block = decoded_block[:-(self.c.K)] # No idea what the extra information is
            decoded += ([1 if k < 0 else 0 for k in decoded_block])

        elif mode == "hard":
            i = to_decode.copy()
            i = 10 * (0.5 - i) # Do weightings

            decoded_block, iters = self.c.decode(i)

            # print("Iterations ", iters)

            decoded_block = decoded_block[:-(self.c.K)] # No idea what the extra information is
            decoded += ([1 if k < 0 else 0 for k in decoded_block])
        else:
            raise Exception("Only 'hard' and 'soft' are valid ldpc decoding modes")
            
        return decoded

    def data_block_processing(self, header = False):
        all_data = []
        actual_data = self.entire_data[self.data_index:]
        ofdm_block_one = actual_data[:self.ofdm_symbol_size+self.ofdm_prefix_size]
        ofdm_block_one = ofdm_block_one[self.ofdm_prefix_size:]
        ideal_block = self.generate_known_ofdm_block()

        assert len(ofdm_block_one) == self.ofdm_symbol_size

        channel_freq = self.channel_estimation(np.fft.fft(ofdm_block_one), ideal_block)

        ofdm_freq = np.fft.fft(ofdm_block_one)

        #sigma2 = self.calculate_sigma2(ofdm_freq, ideal_block)
        

        ofdm_freq = ofdm_freq / channel_freq
        ofdm_freq = ofdm_freq[1:2048]
        corrected = self.combined_correction(ofdm_freq)

        index = 1
        sigma2 = self.calculate_sigma2_one_block(np.fft.fft(ofdm_block_one))

        if header:
            # Extract Header

            ofdm_block_two = actual_data[self.ofdm_symbol_size+self.ofdm_prefix_size: 2 * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block_two = ofdm_block_two[self.ofdm_prefix_size:]

            # sigma2 = self.calculate_sigma2_one_block(ofdm_block_two)

            assert len(ofdm_block_two) == self.ofdm_symbol_size

            data_bins_two, llrs_two = self.ofdm_one_block(ofdm_block_two, sigma2)

            decoded_two = self.ldpc_decode_one_block(data_bins_two, llrs_two)

            restofdata = self.extract_header(decoded_two)

            all_data.extend(restofdata)
            index += 1

        while len(all_data) < self.bits:
            ofdm_block = actual_data[index * (self.ofdm_symbol_size + self.ofdm_prefix_size): (index + 1) * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block = ofdm_block[self.ofdm_prefix_size:]

            #sigma2 = self.calculate_sigma2_one_block(ofdm_block)

            assert len(ofdm_block) == self.ofdm_symbol_size

            data_bins, llrs = self.ofdm_one_block(ofdm_block, sigma2)

            decoded = self.ldpc_decode_one_block(data_bins, llrs)

            all_data.extend(decoded)
            index += 1

        all_data = all_data[:self.bits]
        return all_data
    
    def extract_header(self, data):
        # Extract Header
        data = np.array(data)
        data = np.reshape(data, len(data))

        null_character = [0, 0, 0, 0, 0, 0, 0, 0]

        num_nulls = 0
        name_start = 0
        header_start = 0
        name = []
        header = []
        restofdata = []
        for i in range(0, len(data), 8):
            if (data[i:i+8] == null_character):
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

    def listen(self):
        self.entire_data = np.loadtxt('../recording_2.csv', delimiter = ",", dtype = "float")
        
    def decode_text(self, binary_data):
        binary_data = np.array(binary_data).astype("str")

        ascii = [int(''.join(binary_data[i:i+8]), 2) for i in range(0, len(binary_data), 8)]

        return ''.join([chr(i) for i in ascii])

def success(a, b):
    """find the percentage difference between two lists"""
    successes = 0

    for index, i in enumerate(a):
        if i == b[index]:
            successes += 1 / len(a)

    return successes

from transmitter import transmitter

if __name__ == "__main__":
    t =  transmitter()

    transmitted_bits = t.process_file("max_test_in.txt")

    r = receiver()

    # print(r.decode_text([0, 1, 0, 0, 0, 0, 0, 1]))

    r.set_bits_and_file_name(30704,'asdf')

    r.listen()

    binary_data = r.data_block_processing()

    print(success(binary_data, transmitted_bits))

    print(r.decode_text(binary_data))


### OLD CODE:::


    # def ofdm(self, to_decode, sigma2):
    #     # OLD CODE

    #     decoded_symbols = np.split(to_decode, len(to_decode) / (self.ofdm_prefix_size + self.ofdm_symbol_size))

    #     # Remove Cyclic Prefix

    #     for index, i in enumerate(decoded_symbols):
    #         decoded_symbols[index] = i[self.ofdm_prefix_size:]

    #     # DFT each symbol:

    #     symbols_freq = np.zeros((len(decoded_symbols), self.ofdm_symbol_size)).astype(complex) # 'empty' array

    #     for index, i in enumerate(decoded_symbols):
    #         symbols_freq[index] = np.fft.fft(i)

    #     assert symbols_freq.shape[1] == self.ofdm_symbol_size

    #     # Divide by DFT of Channel Response:

    #     recieved_freq = symbols_freq / self.channel_freq

    #     # Remove complex conjugate bins
    #     constellations = recieved_freq[0][1:2048]
    #     for index, i in enumerate(recieved_freq[1:]):
    #         constellations = np.vstack((constellations, i[1:2048]))

    #     self.constellations = constellations # For showing constellation diagrams

    #     decoded_binary = []
    #     llrs = []

    #     #sigma2 = 1 # Sigma squared - TODO Calculate

    #     # Do Inverse Gray Code:

    #     for symbol in constellations:
    #         for index, i in enumerate(symbol):
    #             if np.real(i) >= 0 and np.imag(i) >= 0:
    #                 decoded_binary.extend([0, 0])
    #             elif np.real(i) <= 0 and np.imag(i) >= 0:
    #                 decoded_binary.extend([0, 1])
    #             elif np.real(i) <= 0 and np.imag(i) <= 0:
    #                 decoded_binary.extend([1, 1])
    #             elif np.real(i) >= 0 and np.imag(i) <= 0:
    #                 decoded_binary.extend([1, 0])
    #             else:
    #                 raise Exception("Gray Code Decoding Error")
                
    #             # Find LLRs by using distance from axes for soft LDPC decoding:

    #             # L_1(y) = c_k \times c_k^* y'_i / \sigma ^ 2

    #             l1 = self.channel_freq[index] * np.conj(self.channel_freq[index]) * np.imag(i) / sigma2

    #             # L_2(y) = c_k \times c_k^* y'_r / \sigma ^ 2 ... from Jossy LDPC Paper

    #             l2 = self.channel_freq[index] * np.conj(self.channel_freq[index]) * np.real(i) / sigma2

    #             # print("Star = ", i, ". l1, l2 =", (l1, l2))
    #             llrs.extend([l1.real, l2.real])

    #     decoded_binary = decoded_binary[:-3296] # TODO find OFDM Padding length automatically
    #     llrs = llrs[:-3296] # TODO find OFDM Padding length automatically

    #     return decoded_binary, llrs

    # def ldpc_decode(self, to_decode, llrs, mode="soft"):
    #     # OLD CODE
    #     decoded = []

    #     print("Number of OFDM Blocks: ", len(llrs))

    #     if mode == "soft":
    #         llrs = np.split(np.array(llrs), len(llrs) // self.c.N)

    #         for i in llrs:
    #             decoded_block, iters = self.c.decode(i)

    #             # print("Iterations ", iters)

    #             decoded_block = decoded_block[:-(self.c.K)] # No idea what the extra information is
    #             decoded += ([1 if k < 0 else 0 for k in decoded_block])
    #     elif mode == "hard":
    #         to_decode = np.split(np.array(to_decode), len(to_decode) // self.c.N)

    #         for i in to_decode:
    #             i = 10 * (0.5 - i) # Do weightings

    #             decoded_block, iters = self.c.decode(i)

    #             # print("Iterations ", iters)

    #             decoded_block = decoded_block[:-(self.c.K)] # No idea what the extra information is
    #             decoded += ([1 if k < 0 else 0 for k in decoded_block])
    #     else:
    #         raise Exception("Only 'hard' and 'soft' are valid ldpc decoding modes")
            
    #     return decoded[:-392] # TODO find LDPC Padding Length automatically