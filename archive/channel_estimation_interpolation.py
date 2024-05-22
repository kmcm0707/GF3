import numpy as np

# received_seqeuence = np.zeros(1497000)
# print(len(received_seqeuence))
'''
make a fake recevied sequence
'''
info_sequence = np.zeros(shape=(1000, 1056))
for x in reversed(range(0, 10)):
    pos = int(x*(1000/10))
    info_sequence = np.insert(info_sequence, pos, np.ones(44100+2))

'''
get the pilot sequeneces from the received sequence
# '''
# pilot_pos = []
# pos = 0
# for x in range(10):
#     pilot_pos.append(pos)
#     pos += 44099+2
#     pilot_pos.append(pos)
#     pos += int(1000/10 * 1056 + 1)

# pilot = np.zeros(shape=(1000, 1056))
# for x in reversed(range(0, 10)):
#     index = 0
#     pos = int(x*(1000/10))
#     pilot = np.insert(pilot, pos, info_sequence[pilot_pos[index]:pilot_pos[index+1]])
#     index += 1

# print(len(pilot[149702:193804]))