import numpy as np
from audio_modem import audio_modem
from scipy import signal
import scipy
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans
from scipy import stats
import math
import matplotlib.pyplot as plt

from transmitter import transmitter

class receiver(audio_modem):
    def __init__(self):
        
        audio_modem.__init__(self)
        self.channel_freq = None
        self.constellations = []
        self.data_index = None
        self.chirp_start = None
        self.past_centers = []
        self.past_angle = 0
        self.past_gradient = 0
        self.past_change = 0
        self.entire_data = None
        self.bits = None
        self.file_name = None
        self.sigma2 = None

        self.t = transmitter() # Used for iterative channel estimation

    def set_bits_and_file_name(self, bits, file_name):
        self.bits = bits
        self.file_name = file_name
    
    def find_start_index(self, data):
        # Find Chirp Start
        # x and y are the time signals to be compared
        x = self.chirp
        cross_correlation = []
        plot_data = data
        data = data[:200000]
        print(len(data))
        n = len(x)
        N = len(data)
        lags = np.arange(-n + 1, N) 
        cross_correlation.append(scipy.signal.correlate(data, x,mode='full', method='fft'))
        cross_correlation = np.array(cross_correlation)
        cross_correlation = uniform_filter1d(cross_correlation, size=5)
        max_index = np.argmax(cross_correlation)
        self.chirp_start = lags[max_index]
        positions = np.arange(0, len(data))
        plt.plot(positions[:lags[max_index] -1024],data[:lags[max_index] -1024])
        plt.plot(positions[lags[max_index] - 1024:lags[max_index] +len(self.chirp_p_s) - 1024],data[lags[max_index] - 1024:lags[max_index] +len(self.chirp_p_s) - 1024], color='r')
        plt.plot(positions[lags[max_index] +len(self.chirp_p_s) - 1024:],data[lags[max_index] +len(self.chirp_p_s) - 1024:], color='g')
        plt.show()
        return lags[max_index], cross_correlation, lags
    
    def channel_estimation(self, block, ideal_block, mode = "basic"):
        """ N.B. doesn't update self.channel_freq!! Find Channel Response - find from received OFDM block and expected (tranmsitted) OFDM block in freq domain """

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

        # self.channel_freq[0] = 1
        # self.channel_freq[self.ofdm_symbol_size // 2] = 1

        # time_freq = np.fft.ifft(self.channel_freq)
        # time_freq = time_freq.real
        # filter = self.ofdm_symbol_size
        # time_freq = time_freq[0:filter]
        # time_freq = np.pad(time_freq, (0, self.ofdm_symbol_size - filter), 'constant', constant_values=(0, 0))
        # channel_freq = np.fft.fft(time_freq)
        # self.channel_freq = channel_freq

        return channel_freq

    def find_data_index(self, data, start_index):
        """ Find Data Start """
        self.data_index = start_index + len(self.chirp_p_s) - self.ofdm_prefix_size
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
        past_change = self.past_change

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
            #print("Centers: ", complex_centers)
            complex_angles = np.angle(complex_centers)
            inti_angles = np.angle(inti_complex)
            """distance = complex_centers - inti_complex"""
            angle_diff = complex_angles - inti_angles
            angle = np.array(angle_diff)
            #print("distance", d)
            #angle = np.angle(d)
            #print("angle", angle)
            """for i in range(len(angle)):
                if angle[i] < 0:
                    angle[i] = 2 * np.pi + angle[i]"""
            angle = np.mean(angle) #- np.pi
            #print("mean angle", angle)
            """if np.abs(angle) > 0.3:
                angle = 0"""
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
        #print("Gradient: ", grad, "Intercept: ", intercept)
        #print("Angle: ", past_angle)
        
        #print("Gradient 2: ", gradient_2)
        """plt.plot(positions, gradient_2)
        plt.show()"""
        
        return corrected

    def ofdm_one_block(self, data_block, sigma2):
        """Decode one block of data - returns (decoded bits - for hard LDPC decoding, LLRs - for soft LDPC decoding)"""
        assert len(data_block) == self.ofdm_symbol_size

        # DFT
        freq = np.fft.fft(data_block)
        
        # Divide by Channel Response
        freq = freq / self.channel_freq

        # Remove Complex Conjugate Bins
        freq = freq[1:2048]

        # Remove Watermark
        watermark = self.generate_known_ofdm_block_mod4()
        # print("Watermark: ", watermark[0:5])

        # Rotate Watermark
        freq = freq * np.exp(-1j * watermark * np.pi / 2)
        corrected = freq
        corrected = self.combined_correction(freq[self.ofdm_bin_min-1:self.ofdm_bin_max])

        self.constellations.extend(corrected)
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
        """ Returns LDPC decoding of one one LDPC block """

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

    def update_channel_estimate(self, received_ofdm_block, decoded_bits):
        generated_ldpc_block = self.t.ldpc_encode(decoded_bits)
        generated_ofdm_block = self.t.ofdm(generated_ldpc_block)[self.ofdm_prefix_size:]

        generated_ofdm_block[0] = 1 # Not sure why this is needed?

        return self.channel_estimation(np.fft.fft(received_ofdm_block), np.fft.fft(generated_ofdm_block), mode="basic")

    def data_block_processing(self, data, header = False):
        """ Main processsing for data block, post synchronisation - 'data' is the data_block"""

        all_data = [] # To be returned

        # actual_data = self.entire_data[self.data_index:]
        # ofdm_block_one = actual_data[:self.ofdm_symbol_size+self.ofdm_prefix_size]
        # ofdm_block_one = ofdm_block_one[self.ofdm_prefix_size:]
        # self.pre_ldpc_data = []

        ofdm_block_one = data[self.ofdm_prefix_size:self.ofdm_symbol_size + self.ofdm_prefix_size]
        self.pre_ldpc_data = [] # For Testing

        ideal_block = self.generate_known_ofdm_block()

        assert len(ofdm_block_one) == self.ofdm_symbol_size

        self.channel_freq = self.channel_estimation(np.fft.fft(ofdm_block_one), ideal_block)
        
        # INITIAL PHASE CORRECTION
        ofdm_freq = np.fft.fft(ofdm_block_one)
        ofdm_freq = ofdm_freq / self.channel_freq
        ofdm_freq = ofdm_freq[1:2048]
        corrected = self.combined_correction(ofdm_freq[self.ofdm_bin_min-1:self.ofdm_bin_max])

        index = 1
        # self.sigma2 = self.calculate_sigma2_one_block(np.fft.fft(ofdm_block_one)) / 2
        self.sigma2 = 1

        if header:
            # Extract Header

            ofdm_block_two = data[self.ofdm_symbol_size+self.ofdm_prefix_size: 2 * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block_two = ofdm_block_two[self.ofdm_prefix_size:]

            # sigma2 = self.calculate_sigma2_one_block(ofdm_block_two)

            assert len(ofdm_block_two) == self.ofdm_symbol_size

            data_bins_two, llrs_two = self.ofdm_one_block(ofdm_block_two, self.sigma2)

            decoded_two = self.ldpc_decode_one_block(data_bins_two, llrs_two)

            restofdata = self.extract_header(decoded_two)

            all_data.extend(restofdata)
            index += 1

        while len(all_data) < self.bits:
            #print("Index: ", index)
            ofdm_block = data[index * (self.ofdm_symbol_size + self.ofdm_prefix_size): (index + 1) * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block = ofdm_block[self.ofdm_prefix_size:] # Remove cyclic prefix

            assert len(ofdm_block) == self.ofdm_symbol_size

            data_bins, llrs = self.ofdm_one_block(ofdm_block, self.sigma2)

            self.pre_ldpc_data.extend(data_bins) # For Testing

            decoded = self.ldpc_decode_one_block(data_bins, llrs)

            all_data.extend(decoded)
            index += 1

            print(self.bin_length, len(decoded))

            updated_channel_estimate = self.update_channel_estimate(ofdm_block, np.array(decoded))
            plt.plot(updated_channel_estimate, color="green", label="Updated channel estimate")
            plt.plot(self.channel_freq, color="red", label="channel estimate before updating")
            plt.title("Updated vs. prior channel estimate( FREQ DOMAIN)")
            plt.legend()
            plt.show()

            plt.plot(np.fft.ifft(updated_channel_estimate), color="green", label="Updated channel estimate")
            plt.plot(np.fft.ifft(self.channel_freq), color="red", label="channel estimate before updating")
            plt.title("Updated vs. prior channel estimate (TIME DOMAIN)")
            plt.legend()
            plt.show()
        all_data = all_data[:self.bits]
        return all_data
    
    def extract_header(self, data):
        """ Extract Header """
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
        # TODO
        self.entire_data = np.loadtxt('../recording_3.csv', delimiter = ",", dtype = "float")
        return self.entire_data
        
    def decode_text(self, binary_data):
        """ Convert binary data to ascii characters """

        binary_data = np.array(binary_data).astype("str")

        ascii = [int(''.join(binary_data[i:i+8]), 2) for i in range(0, len(binary_data), 8)]

        return ''.join([chr(i) for i in ascii])
    
    def start(self):
        data = self.listen()
        start_index, cross_correlation, lags = self.find_start_index(data)
        data_index = self.find_data_index(self.entire_data, start_index)
        return self.data_block_processing(data[data_index:])

def success(a, b):
    """find the percentage difference between two lists"""
    successes = 0

    for index, i in enumerate(a):
        if i == b[index]:
            successes += 1 / len(a)

    return successes

if __name__ == "__main__":
    t =  transmitter()

    transmitted_bits = t.process_file("max_test_in.txt")
    ldpc_bits = t.ldpc_encode(transmitted_bits)

    r = receiver()

    # print(r.decode_text([0, 1, 0, 0, 0, 0, 0, 1]))

    r.set_bits_and_file_name(30704, 'asdf')

    r.listen()

    binary_data = r.start()
    ldpc_bits_r = r.pre_ldpc_data
    print(r.sigma2)

    
    colors = []
    for index in range(0,len(ldpc_bits),2):
        if (ldpc_bits[index], ldpc_bits[index+1]) == (0, 0):
            colors.append('r')
        elif (ldpc_bits[index], ldpc_bits[index+1]) == (0, 1):
            colors.append('g')
        elif (ldpc_bits[index], ldpc_bits[index+1]) == (1, 1):
            colors.append('b')
        elif (ldpc_bits[index], ldpc_bits[index+1]) == (1, 0):
            colors.append('y')
    
    for index in range(29,41):
        
        """plt.scatter(np.real(r.constellations[648 * index:648 * (index+1)]), np.imag(r.constellations[648 * index:648 * (index+1)]), c=colors[648 * index:648 * (index+1)])
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.show()"""

        if False:
            plt.scatter(np.real(r.constellations[648 * index:648 * (index+1)]), np.imag(r.constellations[648 * index:648 * (index+1)]), c=colors[648 * index:648 * (index+1)])
            plt.axhline(0, color='black', lw=0.5)
            plt.axvline(0, color='black', lw=0.5)
            plt.show()

        if index == 30:
            plt.scatter(np.real(r.constellations[648 * index:648 * (index+1)]), np.imag(r.constellations[648 * index:648 * (index+1)]), c=colors[648 * index:648 * (index+1)])
            plt.axhline(0, color='black', lw=0.5)
            plt.axvline(0, color='black', lw=0.5)
            plt.show()

        if index == 40:
            plt.scatter(np.real(r.constellations[648 * index:648 * (index+1)]), np.imag(r.constellations[648 * index:648 * (index+1)]), c=colors[648 * index:648 * (index+1)])
            plt.axhline(0, color='black', lw=0.5)
            plt.axvline(0, color='black', lw=0.5)
            plt.show()



    index = 0
    print(len(ldpc_bits), len(ldpc_bits_r), len(binary_data), len(r.constellations))
    for index in range(0, 40):
        print(success(ldpc_bits[648 * index:648 * (index+1)], ldpc_bits_r[648 * index:648 * (index+1)]))


    print("ldpc")
    for index in range(0, 40):
        print(success(binary_data[648 * index:648 * (index+1)], transmitted_bits[648 * index:648 * (index+1)]))

    #print(r.decode_text(binary_data))

    channel_freq = r.channel_freq
    print(r.chirp_start)


    plt.plot(np.abs(channel_freq))
    plt.show()

    time_freq = np.fft.ifft(channel_freq)
    plt.plot(time_freq)
    plt.show()
    


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