'''
method to estimate channel coefficients through sending known pilot symbols and interpolating the missing freq and time 
(interpolation after sampling)

steps:
1. send pilot symbols of known time and freq of equal amplitudes 
2. upon receiving the symbols, synchronise to ensure correct start of symbols
3. point wise division to retrieve channel gains across pilot time and freq
4. interpolation across time and freq to fill in missing fields (can use ML, deep learning, plane fitting)

Note:
- freq from 1 to 16k
- each pilot symbols sent for 2s
- can be used to combat echo is spaced equally throughout the sent information symbols (accounts for changing channel across time and freq)
- accounts for amplitude attenuation
- taking an instance in discrete time, the channel coeff against freq should look like a rectangular waveform
  Upon DFT, it should give a channel response in discrete time that looks like a sinc

Question
- does not account for freq distortion? (Check with jossy)
- am I just sending sine waves of diff freq at diff time?
- is each pilot symbol sent just amplitude, is it possible to send phase and amplitude and check for phase shift (distortion) by the channel?
'''