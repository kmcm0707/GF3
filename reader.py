data = ''
with open('output.txt', 'r') as file:
    data = file.read().replace('\n', '')

data_arr = list(data)
data_arr = data_arr[:121206*8]
data = ''.join(data_arr)

byte = int(data, 2).to_bytes(121206, 'big')
print(len(data))

output = open("true_output.tiff", "wb")
output.write(byte)
output.close()