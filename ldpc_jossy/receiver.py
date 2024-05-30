import numpy as np
from audio_modem import audio_modem

class receiver(audio_modem):
    def __init__(self):
        
        audio_modem.__init__(self)
        
        self.channel_response()

    def channel_response(self):
        channel_response = np.loadtxt('../weekend_files/channel.csv', delimiter = ",", dtype = "float")
        self.channel_response_padded = np.pad(channel_response, (0, self.ofdm_symbol_size - len(channel_response)))
        self.channel_freq = np.fft.fft(self.channel_response_padded)

        # TODO


    def ofdm(self, to_decode):
        # TODO: Only use bins that are useful

        decoded_symbols = np.split(to_decode, len(to_decode) / (self.ofdm_prefix_size + self.ofdm_symbol_size))

        # Remove Cyclic Prefix

        for index, i in enumerate(decoded_symbols):
            decoded_symbols[index] = i[self.ofdm_prefix_size:]

        # DFT each symbol:

        symbols_freq = np.zeros((len(decoded_symbols), self.ofdm_symbol_size)).astype(complex) # 'empty' array

        for index, i in enumerate(decoded_symbols):
            symbols_freq[index] = np.fft.fft(i)

        assert symbols_freq.shape[1] == self.ofdm_symbol_size

        # Divide by DFT of Channel Response:

        recieved_freq = symbols_freq / self.channel_freq

        # Remove complex conjugate bins
        constellations = recieved_freq[0][1:2048]
        for index, i in enumerate(recieved_freq[1:]):
            constellations = np.vstack((constellations, i[1:2048]))

        self.constellations = constellations # For showing constellation diagrams

        decoded_binary = []
        llrs = []

        sigma2 = 2 # Sigma squared - TODO Calculate

        # Do Inverse Gray Code:

        for symbol in constellations:
            for index, i in enumerate(symbol):
                if np.real(i) >= 0 and np.imag(i) >= 0:
                    decoded_binary.extend([0, 0])
                elif np.real(i) <= 0 and np.imag(i) >= 0:
                    decoded_binary.extend([0, 1])
                elif np.real(i) <= 0 and np.imag(i) <= 0:
                    decoded_binary.extend([1, 1])
                elif np.real(i) >= 0 and np.imag(i) <= 0:
                    decoded_binary.extend([1, 0])
                else:
                    raise Exception("Gray Code Decoding Error")
                
                # Find LLRs by using distance from axes for soft LDPC decoding:

                # L_1(y) = c_k \times c_k^* y'_i / \sigma ^ 2

                l1 = self.channel_freq[index] * np.conj(self.channel_freq[index]) * np.imag(i) / sigma2

                # L_2(y) = c_k \times c_k^* y'_r / \sigma ^ 2 ... from Jossy LDPC Paper

                l2 = self.channel_freq[index] * np.conj(self.channel_freq[index]) * np.real(i) / sigma2

                print("Star = ", i, ". l1, l2 =", (l1, l2))
                llrs.extend([l1.real, l2.real])

        decoded_binary = decoded_binary[:-3296] # TODO find OFDM Padding length automatically
        llrs = llrs[:-3296] # TODO find OFDM Padding length automatically

        return decoded_binary, llrs

    def ldpc_decode(self, to_decode, llrs):
        decoded = []

        llrs = np.split(np.array(llrs), len(llrs) // self.c.N)

        print("Number of OFDM Blocks: ", len(llrs))

        for i in llrs:
            # i = 10 * (0.5 - i) # Do weightings

            decoded_block, iters = self.c.decode(i)

            print("Iterations ", iters)

            decoded_block = decoded_block[:-(self.c.K)] # No idea what the extra information is
            decoded += ([1 if k < 0 else 0 for k in decoded_block])
            
        return decoded[:-392] # TODO find LDPC Padding Length automatically
    
    def decode_text(self, binary_data):
        binary_data = np.array(binary_data).astype("str")

        ascii = [int(''.join(binary_data[i:i+8]), 2) for i in range(0, len(binary_data), 8)]

        return ''.join([chr(i) for i in ascii])
    
if __name__ == "__main__":
    r = receiver()

    print(r.decode_text([0, 1, 0, 0, 0, 0, 0, 1]))