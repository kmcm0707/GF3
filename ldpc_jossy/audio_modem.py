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
        self.chirp = self.generate_chirp(chirp_start, chirp_end, chirp_duration)
        self.chirp_p_s = self.generate_chirp_p_s()
        self.ofdm_symbol_size = ofdm_symbol_size
        self.ofdm_prefix_size = ofdm_prefix_size
        self.sampling_frequency = sampling_frequency
        self.known_ofdm_block_mod4 = self.generate_known_ofdm_block_mod4()
        self.bin_length = ofdm_bin_max - ofdm_bin_min + 1
        self.all_bins = self.ofdm_symbol_size // 2 -1 # 2047
    
    def generate_chirp(self, chirp_start, chirp_end, chirp_duration):
        """Generate Chirp signal with chirp_duration length (number of samples at sampling frequency), from chirp_start to chirp_end frequencies (in terms of bin number)"""
        t = np.linspace(0, int(chirp_duration), int(self.sampling_frequency * chirp_duration), endpoint=False)
        chirp = signal.chirp(t, f0=chirp_start, f1=chirp_end, t1=int(chirp_duration), method='linear').astype(np.float32)
        return chirp

    def generate_known_ofdm_block_mod4(self):
        np.random.seed(1)
        block = np.random.randint(0, 4, size = 2047)
        return block
    
    def generate_chirp_p_s(self):
        """Add a prefix and suffix to the chirp signal"""
        prefix = self.chirp[:-self.ofdm_prefix_size]
        suffix = self.chirp[-self.ofdm_prefix_size:]
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
    

    



