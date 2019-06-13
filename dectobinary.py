def neg_2comp(num,precision):
    x = bin(abs(num))
    b = x[2:]
    for i in range(0,precision-len(b)):#precision-len(b)
        b = '0'+ b

    bin_num=""    
    for c in b:    
        if (c == '0'):
            bin_num = bin_num + '1'
        else:
            bin_num = bin_num + '0'

    bin_num
    x = int(bin_num, 2)
    x = x+1
    y = bin(x)[2:]
    return y

def pos_2comp(num,precision):
    x = bin(abs(num))
    b = x[2:]
    for i in range(0,precision-len(b)):#precision-len(b)
        b = '0'+ b
    return b

def dec2bin(x,precision):
    if(x<0):
       return neg_2comp(x,precision)
    else:
       return pos_2comp(x,precision)
   
def bitstring_to_bytes(s):
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')

def txt2binfile(filename):
    """
    receives txt file with zeros & ones and output binary file
    """
    bin_filename = filename.rsplit( ".", 1 )[ 0 ] + ".dat"
    with open(filename, 'r') as file,open(bin_filename, 'wb') as binfile:
        line = file.readline()
        str_size = len(line)
        #int_size = sys.getsizeof(int())
        #block_size = 8* int_size
        block_size = 8
        mod = str_size % block_size
        if (mod != 0 ):
            line = line + (block_size - mod)* "0"
            str_size = str_size + block_size - mod
        iterations = int(str_size/block_size)
        for i in range(0,iterations):
            #byte = bitstring_to_bytes(line[i*block_size:(i+1)*block_size])
            binfile.write(bitstring_to_bytes(line[i*block_size:(i+1)*block_size]))