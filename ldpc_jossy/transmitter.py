import numpy as np
from audio_modem import audio_modem

class transmitter(audio_modem):
    def __init__(self):
        
        audio_modem.__init__(self)

        self.constellation = None

    def process_file(self, filename):
        with open(filename, 'r', encoding='ascii') as file:
            data = file.read().replace('\n', '')

        binary_string = ''.join(format(ord(x), 'b').zfill(8) for x in data)
        binary_data = np.array(list(binary_string)).astype(int)

        return binary_data
    
    def ldpc_encode(self, binary_data):
        # Pad with zeroes

        padding_zeros = np.zeros(self.c.K - (len(binary_data) % self.c.K))

        print("LDPC Padding Length:", len(padding_zeros))

        to_ldpc = np.concatenate((binary_data, padding_zeros))

        # Split and encode

        coded_binary_data = np.split(to_ldpc, (len(to_ldpc) / self.c.K))

        for index, i in enumerate(coded_binary_data):
            coded_binary_data[index] = self.c.encode(i).astype('int')

        assert len(coded_binary_data[0]) == self.c.N

        return np.array(coded_binary_data).flatten()

    def ofdm(self, to_encode, bin_min=0, bin_max=2048):
        # TODO: Only use bins that are useful

        split_length = self.ofdm_symbol_size - 2 # (symbol_length - 2) as need 511 constellation 'stars' per symbol, and each 'star' represents 2 bits.
        padding_zeros = np.zeros(split_length - (len(to_encode) % split_length))

        print("OFDM Padding Length:", len(padding_zeros))

        to_encode = np.concatenate((to_encode, padding_zeros))

        assert len(to_encode) % split_length == 0.0

        gray_code = np.zeros(len(to_encode) // 2).astype('complex')

        for index, i in enumerate([to_encode[i:i + 2] for i in range(0, len(to_encode), 2)]): # Iterate through two items at a time
            if (i == [0, 0]).all():
                gray_code[index] = 1 + 1j
            elif (i == [0, 1]).all():
                gray_code[index] = -1 + 1j
            elif (i == [1, 1]).all():
                gray_code[index] = -1 - 1j
            elif (i == [1, 0]).all():
                gray_code[index] = 1 - 1j
            else:
                print(i)
                raise Exception("Gray code mapping error")

        self.constellation = gray_code # Used for plotting constellation diagrams

        symbols = np.split(np.array(gray_code), len(gray_code) / (split_length / 2))

        # ENFORCE CONJUGATE SYMMETRY:

        for index, x in enumerate(symbols):
            conj = np.conjugate(x)[::-1]
            symbols[index] = np.concatenate((x, conj), axis=None) # Add reflected conjugate symmetry

            symbols[index] = np.insert(symbols[index], 0, 0)
            symbols[index] = np.insert(symbols[index], int(self.ofdm_symbol_size / 2), 0)

        # Inverse DFT
        info = np.fft.ifft(symbols)

        for i in info[5]:
            assert i.imag == 0 # Check is now real

        # ADD CYCLIC PREFIXES

        to_transmit = np.zeros(shape = (len(info), self.ofdm_symbol_size + self.ofdm_prefix_size))

        for index, x in enumerate(info):
            cyclic_prefix = x[-self.ofdm_prefix_size:]
            to_transmit[index]  = np.concatenate((cyclic_prefix, x), axis = None)

        # print(to_transmit.shape) # Should be (~~, 4096 + 1024)

        to_transmit = np.concatenate(to_transmit, axis = 0)

        return to_transmit

    
    def play_sound(self):
        # TODO
        pass

    # TODO: Prepend chirp and known OFDM Block

    def transmit(self, filename):
        binary_data = self.process_file(filename)
        coded_binary_data = self.ldpc_encode(binary_data)
        to_transmit = self.ofdm(coded_binary_data)

        return to_transmit


if __name__ == "__main__":
    t = transmitter()

    t.transmit("max_test_in.txt")

