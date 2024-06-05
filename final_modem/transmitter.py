import numpy as np
from audio_modem import audio_modem
import pyaudio
import time
import pandas as pd

class transmitter(audio_modem):
    def __init__(self):
        
        audio_modem.__init__(self)


    def process_file(self, filename: str):
        """open file and return bits as numpy array"""

        with open(filename, 'rb') as f:
            bytes=f.read()
        return np.unpackbits(np.frombuffer(bytes, dtype=np.uint8))
    
    def ldpc_encode(self, binary_data):
        """binary_data: numpy array of binary data to be encoded"""
        print("LDPC Encoding Length:", len(binary_data))
        # Pad with zeroes
        padding_zeros = []
        if len(binary_data) % self.c.K != 0:
            padding_zeros = np.zeros(self.c.K - (len(binary_data) % self.c.K))
        
        print("LDPC Padding Length:", len(padding_zeros))

        if len(padding_zeros) != 0:
            to_ldpc = np.concatenate((binary_data, padding_zeros))
        else:
            to_ldpc = binary_data
        # Split and encode

        coded_binary_data = np.split(to_ldpc, (len(to_ldpc) / self.c.K))

        for index, i in enumerate(coded_binary_data):
            coded_binary_data[index] = self.c.encode(i).astype('int')

        assert len(coded_binary_data[0]) == self.c.N

        return np.array(coded_binary_data).flatten()

    def ofdm(self, to_encode):
        """ofdm encode a block"""

        print("OFDM Encoding Length:", len(to_encode))
        split_length = (self.bin_length) * 2
        padding_zeros = []
        if len(to_encode) % split_length != 0:
            padding_zeros = np.zeros(split_length - (len(to_encode) % split_length))

        print("OFDM Padding Length:", len(padding_zeros))

        if len(padding_zeros) != 0:
            to_encode = np.concatenate((to_encode, padding_zeros))

        assert len(to_encode) % split_length == 0.0

        to_encode_split = np.split(to_encode, len(to_encode) / split_length)
        un_watermarked_data = np.zeros(len(to_encode) // 2).astype('complex')
        watermark = self.generate_known_ofdm_block_mod4()
        data_all = []
        for index, i in enumerate(to_encode_split):
            mod4_i = self.binary_symbol_to_mod4(i)
            for ii in range(len(mod4_i)):
                un_watermarked_data[ii + index * split_length // 2] = self.mod4_to_gray(mod4_i[ii])
            padded_i = np.pad(mod4_i, (self.ofdm_bin_min -1, self.ofdm_symbol_size // 2 - self.ofdm_bin_max - 1), 'constant', constant_values=(0, 0))
            assert len(padded_i) == self.all_bins
            mod4_added = self.mod_4_addition(padded_i, watermark)
            data = np.zeros(self.all_bins).astype('complex')
            for iii in range(len(mod4_added)):
                data[iii] = self.mod4_to_gray(mod4_added[iii])
            data_all.append(data)

        data_all = np.array(data_all)
        assert data_all.shape[1] == self.all_bins

        self.constellation = un_watermarked_data

        # ENFORCE CONJUGATE SYMMETRY:
        symbols = np.zeros((len(data_all), self.ofdm_symbol_size)).astype('complex')
        for index, x in enumerate(data_all):
            conj = np.conjugate(x)[::-1]
            temp = np.concatenate((x, conj), axis=None)
            temp = np.insert(temp, 0, 0)
            temp = np.insert(temp, int(self.ofdm_symbol_size / 2), 0)
            symbols[index] = temp


        assert symbols.shape[1] == self.ofdm_symbol_size

        # Inverse DFT
        info = np.fft.ifft(symbols)
        assert info.shape[1] == self.ofdm_symbol_size
        for i in info[0]:
            assert i.imag == 0
        to_transmit = np.zeros(shape = (len(info), self.ofdm_symbol_size + self.ofdm_prefix_size))
        for index, x in enumerate(info):
            cyclic_prefix = x[-self.ofdm_prefix_size:]
            to_transmit[index]  = np.concatenate((cyclic_prefix, x), axis = None)
        
        assert to_transmit.shape[1] == self.ofdm_symbol_size + self.ofdm_prefix_size

        to_transmit = np.concatenate(to_transmit, axis = 0)
        return to_transmit
    
    def assemble_all(self, to_transmit, chirp_p_s, known_ofdm_cp_ifft, mode="five"):
        if mode == "five":
            return np.concatenate((chirp_p_s, known_ofdm_cp_ifft, known_ofdm_cp_ifft, known_ofdm_cp_ifft, known_ofdm_cp_ifft, known_ofdm_cp_ifft, to_transmit, chirp_p_s), axis = None)
        elif mode == "one":
            return np.concatenate((chirp_p_s, known_ofdm_cp_ifft, to_transmit, chirp_p_s), axis = None)
    
    def play_sound(self, samples):
        p = pyaudio.PyAudio()
        time.sleep(1)
        samples = samples.astype(np.float32)
        samples = samples / np.max(np.abs(samples))
        output_bytes = (1 * samples).tobytes()
        # for paFloat32 sample values must be in range [-1.0, 1.0]
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.sampling_frequency,
                        output=True)

        # play. May repeat with different volume values (if done interactively)
        start_time = time.time()
        stream.write(output_bytes)
        print("Played sound for {:.2f} seconds".format(time.time() - start_time))

        stream.stop_stream()
        stream.close()

        p.terminate()

    def transmit(self, filename, playsound=True):
        binary_data = self.process_file(filename)
        binary_data_with_header = self.add_header(binary_data)
        print(len(binary_data_with_header))
        coded_binary_data = self.ldpc_encode(binary_data_with_header)
        to_transmit = self.ofdm(coded_binary_data)
        chirp_p_s = self.chirp_p_s * 0.1
        print(len(chirp_p_s))
        known_ofdm_cp_ifft = self.generate_known_ofdm_block_cp_ifft()
        to_transmit = self.assemble_all(to_transmit, chirp_p_s, known_ofdm_cp_ifft)
        print(len(to_transmit))
        if playsound:
            self.play_sound(to_transmit)
        return to_transmit

    def add_header(self, binary_data):
        null_character = np.zeros(8).astype(int)
        bits = np.unpackbits(np.frombuffer(b"56840", dtype=np.uint8))
        file_name = np.unpackbits(np.frombuffer(b"hamlet.txt", dtype=np.uint8))
        return np.concatenate((null_character, null_character, file_name, null_character, null_character, bits, null_character, null_character, binary_data))

if __name__ == "__main__":
    t = transmitter()
    print(t.transmit("data/hamlet_in.txt", False))

