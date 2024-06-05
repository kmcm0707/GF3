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
        self.chirp_start = None
        self.past_centers = []
        self.past_angle = 0
        self.past_gradient = 0
        self.past_change = 0
        self.entire_data = None
        self.bits = None
        self.file_name = None
        self.sigma2 = None
        self.pre_ldpc_data = []
        self.corrected = []
        self.first_decoded = []
        self.times = 0

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
        cross_correlation = np.reshape(cross_correlation, cross_correlation.shape[1])
        print(cross_correlation.shape)
        plt.plot(cross_correlation)
        plt.show()
        cross_correlation = np.abs(cross_correlation)
        cross_correlation = uniform_filter1d(cross_correlation, size=5)
        plt.plot(cross_correlation)
        plt.show()
        max_index = np.argmax(cross_correlation)
        self.chirp_start = lags[max_index]
        positions = np.arange(0, len(data))
        plt.plot(positions[:lags[max_index] -1024],data[:lags[max_index] -1024])
        plt.plot(positions[lags[max_index] - 1024:lags[max_index] +len(self.chirp_p_s) - 1024],data[lags[max_index] - 1024:lags[max_index] +len(self.chirp_p_s) - 1024], color='r')
        plt.plot(positions[lags[max_index] +len(self.chirp_p_s) - 1024:],data[lags[max_index] +len(self.chirp_p_s) - 1024:], color='g')
        plt.show()
        return lags[max_index], cross_correlation, lags
    
    def find_end_index(self, data):
        # Find Chirp End
        # x and y are the time signals to be compared
        x = self.chirp_p_s
        cross_correlation = []
        plot_data = data
        data = data[300000:]
        print(len(data))
        n = len(x)
        N = len(data)
        lags = np.arange(-n + 1, N) 
        cross_correlation.append(scipy.signal.correlate(data, x,mode='full', method='fft'))
        cross_correlation = np.array(cross_correlation)
        cross_correlation = np.reshape(cross_correlation, cross_correlation.shape[1])
    
        cross_correlation = np.abs(cross_correlation)
        cross_correlation = uniform_filter1d(cross_correlation, size=5)

        max_index = np.argmax(cross_correlation)
        self.chirp_start = lags[max_index]
        positions = np.arange(0, len(data))
        
        return lags[max_index] + 300000, cross_correlation, lags
    
    def channel_estimation(self, block, ideal_block):
        # Find Channel Response
        # H = Y/X
        # Y = Recieved OFDM Blocks
        # X = Ideal OFDM Blocks
        # H = Channel Response
        self.channel_freq = np.true_divide(block, ideal_block, out=np.zeros_like(block), where=ideal_block!=0).astype(complex)
        self.channel_freq[0] = 1
        self.channel_freq[self.ofdm_symbol_size // 2] = 1

        time_freq = np.fft.ifft(self.channel_freq)
        time_freq = time_freq.real
        filter = self.ofdm_symbol_size
        time_freq = time_freq[0:filter]
        time_freq = np.pad(time_freq, (0, self.ofdm_symbol_size - filter), 'constant', constant_values=(0, 0))
        channel_freq = np.fft.fft(time_freq)
        self.channel_freq = channel_freq
        return self.channel_freq

    def find_data_index(self, data, start_index):
        # Find Data Start
        self.data_index = start_index + len(self.chirp_p_s) - self.ofdm_prefix_size
        return self.data_index
    
    def calculate_sigma2_five_block(self, recieved, ideal):
        ideal = ideal * self.channel_freq
        ideal = np.concatenate(ideal)
        recieved = np.concatenate(recieved)
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

    def calculate_sigma2(self, recieved, ideal):
        # TODO: CHECK WITH MAX CORRECT
        # Calculate Noise Power
        # sigma2 = 1/N * \sum_{k=1}^{N} |Y_k - H_kX_k|^2
        # Y = Recieved OFDM Blocks
        # X = Ideal OFDM Blocks
        # H = Channel Response
        # sigma2 = Noise Power
        ideal = ideal * self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1]
        ideal = np.concatenate(ideal)
        recieved = np.concatenate(recieved)
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
        """Decode one block of data"""
        assert len(data_block) == self.ofdm_symbol_size

        # DFT
        freq = np.fft.fft(data_block)
        
        # Divide by Channel Response
        freq = freq / self.channel_freq

        # Remove Complex Conjugate Bins
        freq = freq[1:2048]

        # Remove Watermark
        watermark = self.generate_known_ofdm_block_mod4()
        watermark = watermark
        # print("Watermark: ", watermark[0:5])

        # Rotate Watermark
        freq = freq * np.exp(-1j * watermark * np.pi / 2)
        corrected = freq
        corrected = self.combined_correction(freq[self.ofdm_bin_min-1:self.ofdm_bin_max])
        corrected = corrected
        self.corrected.append(corrected * self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1])

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

    def ofdm_one_block_2(self, data_block, sigma2):
        data_block = np.array(data_block)
        data_block = np.reshape(data_block, len(data_block))

        assert len(data_block) == self.bin_length

        data_block = data_block / self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1]

        self.corrected.append(data_block * self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1])
        self.constellations.extend(data_block)

        decoded = []
        llrs = []
        for index, i in enumerate(data_block):
            decoded.extend(self.constellation_point_to_binary(i))

            l1 = self.channel_freq[index + self.ofdm_bin_min] * np.conj(self.channel_freq[index + self.ofdm_bin_min]) * np.imag(i) / sigma2
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
            decoded_temp = []
            decoded_temp += ([1 if k < 0 else 0 for k in decoded_block])
            self.first_decoded.append(decoded_temp)
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

    def data_block_processing(self, five_blocks = False):
        all_data = []
        actual_data = self.entire_data[self.data_index:]
        ofdm_block_one = actual_data[:self.ofdm_symbol_size+self.ofdm_prefix_size]
        ofdm_block_one = ofdm_block_one[self.ofdm_prefix_size:]
        self.pre_ldpc_data = []
        ideal_block = self.generate_known_ofdm_block()

        assert len(ofdm_block_one) == self.ofdm_symbol_size

        channel_freq = self.channel_estimation(np.fft.fft(ofdm_block_one), ideal_block)
        self.channel_freq = channel_freq

        index = 1
        if five_blocks == False:
            ofdm_freq = np.fft.fft(ofdm_block_one)
            self.sigma2 = self.calculate_sigma2_one_block(np.fft.fft(ofdm_block_one)) / 2
            ofdm_freq = ofdm_freq / channel_freq
            ofdm_freq = ofdm_freq[1:2048]
            corrected = self.combined_correction(ofdm_freq[self.ofdm_bin_min-1:self.ofdm_bin_max])
        else:
            ofdm_block_two = actual_data[self.ofdm_symbol_size+self.ofdm_prefix_size: 2 * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block_two = ofdm_block_two[self.ofdm_prefix_size:]
            ofdm_block_three = actual_data[2 * (self.ofdm_symbol_size + self.ofdm_prefix_size): 3 * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block_three = ofdm_block_three[self.ofdm_prefix_size:]
            ofdm_block_four = actual_data[3 * (self.ofdm_symbol_size + self.ofdm_prefix_size): 4 * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block_four = ofdm_block_four[self.ofdm_prefix_size:]
            ofdm_block_five = actual_data[4 * (self.ofdm_symbol_size + self.ofdm_prefix_size): 5 * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block_five = ofdm_block_five[self.ofdm_prefix_size:]

            channel_freq_2 = self.channel_estimation(np.fft.fft(ofdm_block_two), ideal_block)
            channel_freq_3 = self.channel_estimation(np.fft.fft(ofdm_block_three), ideal_block)
            channel_freq_4 = self.channel_estimation(np.fft.fft(ofdm_block_four), ideal_block)
            channel_freq_5 = self.channel_estimation(np.fft.fft(ofdm_block_five), ideal_block)

            channel_freq = (channel_freq + channel_freq_2 + channel_freq_3 + channel_freq_4 + channel_freq_5) / 5
            self.channel_freq = channel_freq

            ofdm_freq_1 = np.fft.fft(ofdm_block_one)
            ofdm_freq_2 = np.fft.fft(ofdm_block_two)
            ofdm_freq_3 = np.fft.fft(ofdm_block_three)
            ofdm_freq_4 = np.fft.fft(ofdm_block_four)
            ofdm_freq_5 = np.fft.fft(ofdm_block_five)

            ofdm_freq_1 = ofdm_freq_1 / channel_freq
            ofdm_freq_2 = ofdm_freq_2 / channel_freq
            ofdm_freq_3 = ofdm_freq_3 / channel_freq
            ofdm_freq_4 = ofdm_freq_4 / channel_freq
            ofdm_freq_5 = ofdm_freq_5 / channel_freq

            ofdm_freq_1 = ofdm_freq_1[1:2048]
            ofdm_freq_2 = ofdm_freq_2[1:2048]
            ofdm_freq_3 = ofdm_freq_3[1:2048]
            ofdm_freq_4 = ofdm_freq_4[1:2048]
            ofdm_freq_5 = ofdm_freq_5[1:2048]

            corrected_1 = self.combined_correction(ofdm_freq_1[self.ofdm_bin_min-1:self.ofdm_bin_max])
            corrected_2 = self.combined_correction(ofdm_freq_2[self.ofdm_bin_min-1:self.ofdm_bin_max])
            corrected_3 = self.combined_correction(ofdm_freq_3[self.ofdm_bin_min-1:self.ofdm_bin_max])
            corrected_4 = self.combined_correction(ofdm_freq_4[self.ofdm_bin_min-1:self.ofdm_bin_max])
            corrected_5 = self.combined_correction(ofdm_freq_5[self.ofdm_bin_min-1:self.ofdm_bin_max])

            ideal_block = self.generate_known_ofdm_block()
            self.sigma2 = self.calculate_sigma2_five_block([corrected_1, corrected_2, corrected_3, corrected_4, corrected_5], [ideal_block, ideal_block, ideal_block, ideal_block, ideal_block])
            index = 5

        
        while len(all_data) < self.bits:
            #print("\n")
            #print("Index: ", index)
            ofdm_block = actual_data[index * (self.ofdm_symbol_size + self.ofdm_prefix_size): (index + 1) * (self.ofdm_symbol_size + self.ofdm_prefix_size)]
            ofdm_block = ofdm_block[self.ofdm_prefix_size:]

            #sigma2 = self.calculate_sigma2_one_block(ofdm_block)

            assert len(ofdm_block) == self.ofdm_symbol_size

            data_bins, llrs = self.ofdm_one_block(ofdm_block, self.sigma2)

            self.pre_ldpc_data.extend(data_bins)

            decoded = self.ldpc_decode_one_block(data_bins, llrs)

            all_data.extend(decoded)
            index += 1
        
        self.corrected = np.array(self.corrected)
        self.first_decoded = np.array(self.first_decoded)
        return self.data_block_processing_part_2()

        #all_data = all_data[:self.bits]
        #return all_data
    
    def data_block_processing_part_2(self):
        corrected = self.corrected.copy()
        first_decoded = self.first_decoded.copy()
        first_decoded_constellations = []

        for i in first_decoded:
            first_decoded_constellations.append(self.binary_to_constellation_point(i))
        
        first_decoded_constellations = np.array(first_decoded_constellations)

        temp_channels = []
        for index, i in enumerate(corrected):
            #print(index, i.shape, first_decoded_constellations[index].shape)
            temp_channels.append(i / first_decoded_constellations[index])
        
        temp_channels = np.array(temp_channels)

        mean = np.mean(temp_channels, axis=0)

        plt.plot(np.abs(self.channel_freq))

        self.channel_freq[self.ofdm_bin_min:self.ofdm_bin_max+1] = mean

        plt.plot(np.abs(self.channel_freq), color='r')
        plt.show()

        self.sigma2 = self.calculate_sigma2(corrected, first_decoded_constellations)

        self.pre_ldpc_data = []
        self.corrected = []
        self.first_decoded = []
        self.all_data = []
        self.constellations = []
        
        for index, i in enumerate(corrected):
            data_bins, llrs = self.ofdm_one_block_2(i, self.sigma2)
            self.pre_ldpc_data.extend(data_bins)

            decoded = self.ldpc_decode_one_block(data_bins, llrs)

            self.all_data.extend(decoded)

        self.corrected = np.array(self.corrected)
        self.first_decoded = np.array(self.first_decoded)

        all_data = self.all_data
        self.times += 1
        if self.times == 5:
            return all_data
        else:
            return self.data_block_processing_part_2()
     
    def extract_header(self, data):
        # Extract Header
        data = np.array(data)
        data = np.reshape(data, len(data))

        null_character = [0, 0, 0, 0, 0, 0, 0, 0]
        null_character = np.array(null_character)

        num_nulls = 0
        name_start = 0
        header_start = 0
        name = []
        header = []
        restofdata = []
        for i in range(0, len(data), 8):
            if ((data[i:i+8] == null_character).all()):
                print(i)
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
        #print("Name: ", name)
        #print("Header: ", header)
        file_name = self.decode_text(name)
        bits = self.decode_text(header)

        print("File Name: ", file_name)
        print("Header: ", bits)

        self.set_bits_and_file_name(bits, file_name)

        return restofdata

    def listen(self):
        self.entire_data = np.loadtxt('../malachy_testing3.csv', delimiter = ",", dtype = "float")
        
    def decode_text(self, binary_data):
        binary_data = np.array(binary_data).astype("str")

        ascii = [int(''.join(binary_data[i:i+8]), 2) for i in range(0, len(binary_data), 8)]

        return ''.join([chr(i) for i in ascii])
    
    def save_decoded_file(self, data_arr, size):
        data = ''.join(data_arr)
        byte = int(data, 2).to_bytes(size, 'big')
        print(len(data))
        output = open(self.file_name, "wb")
        output.write(byte)
        output.close()

    def start(self):
        self.listen()
        start_index, cross_correlation, lags = self.find_start_index(self.entire_data)
        end_index, cross_correlation, lags = self.find_end_index(self.entire_data)
        end_index = end_index - self.ofdm_prefix_size
        data_index = self.find_data_index(self.entire_data, start_index)
        data_length = end_index - data_index
        num_blocks = data_length // (self.ofdm_symbol_size + self.ofdm_prefix_size) + data_length % (self.ofdm_symbol_size + self.ofdm_prefix_size)
        self.bits = num_blocks * self.c.K
        print("num_blocks: ", num_blocks)
        self.bits = 30704
        print("bits: ", self.bits)
        data = self.data_block_processing()
        print("Data: ", data[0:100])
        all_data = self.extract_header(data)
        all_data = all_data[:self.bits]
        self.save_decoded_file(all_data, self.bits)
        return all_data

def success(a, b):
    """find the percentage difference between two lists"""
    successes = 0
    wrong_indices = []
    for index, i in enumerate(a):
        if i == b[index]:
            successes += 1 / len(a)
        else:
            wrong_indices.append(index)
    #print("Wrong Indices: ", wrong_indices)
    return successes

from transmitter import transmitter

if __name__ == "__main__":
    t =  transmitter()

    transmitted_bits = t.process_file("max_test_in.txt")
    ldpc_bits = t.ldpc_encode(transmitted_bits)

    r = receiver()

    # print(r.decode_text([0, 1, 0, 0, 0, 0, 0, 1]))

    r.set_bits_and_file_name(30704,'asdf')

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

    print("first decoded", r.first_decoded.shape)
    print("corrected", r.corrected.shape)

    index = 0
    print(len(ldpc_bits), len(ldpc_bits_r), len(binary_data), len(r.constellations))
    for index in range(0, 40):
        print(success(ldpc_bits[648 * 2 * index:648 * 2 * (index+1)], ldpc_bits_r[648 * 2 * index: 648 * 2 * (index+1)]))


    print("ldpc")
    for index in range(0, 40):
        print(success(binary_data[648 * index:648 * (index+1)], transmitted_bits[648 * index:648 * (index+1)]))

    print(r.decode_text(binary_data))

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