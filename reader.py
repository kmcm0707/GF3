data = ''
with open('edit7.txt', 'r') as file:
    data = file.read().replace('\n', '')

size = 122790
data_arr = list(data)
data_arr = data_arr[:size*8]
data = ''.join(data_arr)

byte = int(data, 2).to_bytes(size, 'big')
print(len(data))

output = open("true_output7.tiff", "wb")
output.write(byte)
output.close()