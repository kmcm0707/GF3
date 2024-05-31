import platform

if platform.system() == "Darwin":
    from py import ldpc
else:
    from pys import ldpc

class audio_modem:
    def __init__(self,
                 sampling_frequency = 48000,
                 chirp_start = 65, # Bin number
                 chirp_end = 753, # Bin number
                 chirp_duration = 65536, # Number of samples
                 ofdm_symbol_size = 4096,
                 ofdm_prefix_size = 1024,
                 ldpc_standard = "802.16", 
                 ldpc_rate = "1/2",
                 ldpc_z = 54):
        
        self.c = ldpc.code(standard = ldpc_standard, rate = ldpc_rate, z = ldpc_z)
        self.chirp = self.generate_chirp(chirp_start, chirp_end, chirp_duration)

        self.ofdm_symbol_size = ofdm_symbol_size
        self.ofdm_prefix_size = ofdm_prefix_size
        self.sampling_frequency = sampling_frequency
    
    def generate_chirp(self, chirp_start, chirp_end, chirp_duration):
        """Generate Chirp signal with chirp_duration length (number of samples at sampling frequency), from chirp_start to chirp_end frequencies (in terms of bin number)"""
        
        # TODO

        return None