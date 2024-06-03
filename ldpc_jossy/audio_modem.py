import platform
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd


if platform.system() == "Darwin":
    from py import ldpc
else:
    from pys import ldpc

class audio_modem:
    def __init__(self,
                 sampling_frequency = 48000,
                 chirp_start = 761.72, # Frequency in Hz
                 chirp_end = 8824.22, # Frequency in Hz
                 chirp_duration = 1.365, # Time in seconds
                 ofdm_symbol_size = 4096,
                 ofdm_prefix_size = 1024,
                 ofdm_bin_min = 85,
                 ofdm_bin_max = 85 + 648 - 1,
                 ldpc_standard = "802.16", 
                 ldpc_rate = "1/2",
                 ldpc_z = 54):
        
        self.c = ldpc.code(standard = ldpc_standard, rate = ldpc_rate, z = ldpc_z)
        self.sampling_frequency = sampling_frequency
        self.ofdm_symbol_size = ofdm_symbol_size
        self.ofdm_prefix_size = ofdm_prefix_size
        self.bin_length = ofdm_bin_max - ofdm_bin_min + 1
        self.all_bins = self.ofdm_symbol_size // 2 -1 # 2047
        self.ofdm_bin_min = ofdm_bin_min
        self.ofdm_bin_max = ofdm_bin_max
        self.known_ofdm_block_mod4 = self.generate_known_ofdm_block_mod4()
        self.known_ofdm_block = self.generate_known_ofdm_block()
        

        
        self.chirp = self.generate_chirp(chirp_start, chirp_end, chirp_duration)
        self.chirp_p_s = self.generate_chirp_p_s()
    
    def generate_chirp(self, chirp_start, chirp_end, chirp_duration):
        """Generate Chirp signal with chirp_duration length (number of samples at sampling frequency), from chirp_start to chirp_end frequencies (in terms of bin number)"""
        t = np.linspace(0, int(chirp_duration), int(self.sampling_frequency * chirp_duration), endpoint=False)
        chirp = signal.chirp(t, f0=chirp_start, f1=chirp_end, t1=int(chirp_duration), method='linear').astype(np.float32)
        return chirp

    def generate_known_ofdm_block_mod4(self):
        np.random.seed(1)
        block = np.random.randint(0, 4, size = self.all_bins)
        return block
    
    def generate_known_ofdm_block(self):
        """Generate known OFDM block"""
        known_ofdm_block_mod4 = self.generate_known_ofdm_block_mod4()
        
        known_OFDM_constellation = np.zeros(self.all_bins).astype('complex')
        for i in range(len(known_ofdm_block_mod4)):
            known_OFDM_constellation[i] = self.mod4_to_gray(known_ofdm_block_mod4[i])

        total_block = np.concatenate((known_OFDM_constellation, np.conjugate(known_OFDM_constellation)[::-1]))
        total_block = np.insert(total_block, 0, 0)
        total_block = np.insert(total_block, self.ofdm_symbol_size // 2, 0)

        assert len(total_block) == self.ofdm_symbol_size

        return total_block
         
    def generate_known_ofdm_block_cp_ifft(self):
        """Generate known OFDM block with cyclic prefix and IFFT"""
        known_ofdm_block_mod4 = self.generate_known_ofdm_block_mod4()

        known_OFDM_constellation = np.zeros(self.all_bins).astype('complex')
        for i in range(len(known_ofdm_block_mod4)):
            known_OFDM_constellation[i] = self.mod4_to_gray(known_ofdm_block_mod4[i])

        total_block = np.concatenate((known_OFDM_constellation, np.conjugate(known_OFDM_constellation)[::-1]))
        total_block = np.insert(total_block, 0, 0)
        total_block = np.insert(total_block, self.ofdm_symbol_size // 2, 0)

        assert len(total_block) == self.ofdm_symbol_size

        known_OFDM_block = np.fft.ifft(total_block)
        assert known_OFDM_block[10].imag == 0

        cyclic_prefix = known_OFDM_block[-self.ofdm_prefix_size:]
        known_OFDM_block_cp_ifft = np.concatenate((cyclic_prefix, known_OFDM_block), axis = None)

        return known_OFDM_block_cp_ifft
        
    def generate_chirp_p_s(self):
        """Add a prefix and suffix to the chirp signal"""
        prefix = self.chirp[-self.ofdm_prefix_size:]
        print("Chirp_p_s Prefix Length:", len(prefix))
        suffix = self.chirp[:self.ofdm_prefix_size]
        print("Chirp_p_s Suffix Length:", len(suffix))
        print("Chirp_p_s Length:", len(np.concatenate((prefix, self.chirp, suffix))))
        return np.concatenate((prefix, self.chirp, suffix))
    
    def mod4_to_gray(self, mod4):
        """Converts a mod4 symbol to a Gray code symbol"""
        if mod4 == 0:
            return 1 + 1j
        elif mod4 == 1:
            return -1 + 1j
        elif mod4 == 2:
            return -1 - 1j
        elif mod4 == 3:
            return 1 - 1j
        else:
            raise ValueError("mod4 symbol must be in range [0, 3]")
        
    def mod_4_addition(self, a, b):
        """Adds two mod4 symbols"""
        return (a + b) % 4
    
    def binary_symbol_to_mod4(self, binary_symbol):
        """Converts a binary symbol to a mod4 symbol"""
        mod4_symbol = []
        for i in range(0, len(binary_symbol) - 1, 2):
            if (binary_symbol[i], binary_symbol[i+1]) == (0, 0):
                mod4_symbol.append(0)
            elif (binary_symbol[i], binary_symbol[i+1]) == (0, 1):
                mod4_symbol.append(1)
            elif (binary_symbol[i], binary_symbol[i+1]) == (1, 1):
                mod4_symbol.append(2)
            elif (binary_symbol[i], binary_symbol[i+1]) == (1, 0):
                mod4_symbol.append(3)      
        return mod4_symbol
    
    def binary_to_constellation_point(self, binary_symbol):
        """Converts a binary symbol to a constellation point"""
        #print(binary_symbol)
        mod4_symbol = self.binary_symbol_to_mod4(binary_symbol)
        #print("Mod4 Symbol:", len(mod4_symbol))
        constellation_point = np.zeros(len(mod4_symbol)).astype('complex')
        for i in range(len(mod4_symbol)):
            constellation_point[i] = self.mod4_to_gray(mod4_symbol[i])
        return constellation_point
    
    def constellation_point_to_binary(self, constellation_point):
        """Converts a constellation point to a binary symbol"""

        if np.real(constellation_point) >= 0 and np.imag(constellation_point) >= 0:
            return [0, 0]
        elif np.real(constellation_point) <= 0 and np.imag(constellation_point) >= 0:
            return [0, 1]
        elif np.real(constellation_point) <= 0 and np.imag(constellation_point) <= 0:
            return [1, 1]
        elif np.real(constellation_point) >= 0 and np.imag(constellation_point) <= 0:
            return [1, 0]
        else:
            raise Exception("Gray Code Decoding Error")

        # if (constellation_point == 1 + 1j).all():
        #     return [0, 0]
        # elif (constellation_point == -1 + 1j).all():
        #     return [0, 1]
        # elif (constellation_point == -1 - 1j).all():
        #     return [1, 1]
        # elif (constellation_point == 1 - 1j).all():
        #     return [1, 0]
        # else:
        #     raise ValueError("constellation point must be in range [1 + 1j, -1 - 1j]")
             
    

    



