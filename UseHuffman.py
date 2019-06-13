#!/usr/bin/env python

import sys
from huffman import HuffmanCoding
import os
import numpy as np


    
#def datToBin(datfile):
#    """this function accepts .dat file and retutn binary file"""
#    with open(datfile, 'r') as binary_file:
#        #binary_file.replace('\n','')
#        data = binary_file.read()
#        binarydata = ''.join(format(ord(x), 'b') for x in data)
#    filename, file_extension = os.path.splitext(datfile)
#    binfile = filename + ".bin" 
#    with open(binfile, 'w') as binary_file:
#        binary_file.write(binarydata)
#    return binfile
    
def datToBin(datfile):
    """this function accepts .dat file and retutn binary file"""
    x = np.fromfile(datfile , dtype = np.uint8 , count= -1,sep ='')
#    c = bytearray(x)
    b = []
    st = []
    for i in range(len(x)):
       b.append(bin(x[i]))
       st.append(str(b[i]))
    for i in range(len(st)): 
        st[i]=st[i][2:]
    binarydata = ''.join(st)
    filename, file_extension = os.path.splitext(datfile)
 #   binfile = filename + ".bin" 
#    with open(binfile, 'wb') as binary_file:
#        binary_file.write(c)
    binfile = filename + ".txt" 
    with open(binfile, 'w') as binary_file:
        binary_file.write(binarydata)
    return binfile

#with open(filename, 'wb') as binfile:
#    binfile.write(b'\x00' * desired_length)

#def use_huffman(filename, wordlength= 14):
#    binFilePath = datToBin(filename)
#    h = HuffmanCoding(binFilePath,wordlength)
#    output_path = h.compress()
#    #h.decompress(output_path)


def use_huffman(filename, wordlength= 14):
    h = HuffmanCoding(filename,wordlength)
    output_path = h.compress()
    #h.decompress(output_path)
    return output_path
    

