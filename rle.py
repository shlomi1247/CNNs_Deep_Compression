# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:48:44 2019

@author: shlom

RLE for DeepCompression
"""

#!/usr/bin/python

from math import log,ceil
import numpy as np
import os


log2 = lambda x: log(x,2)


def binary(x,l=1):
	fmt = '{0:0%db}' % l
	return fmt.format(x)

def unary(x):
	return x*'0'+'1'

def elias_gamma(x):
    if x == 0: return '0'
    if x == 1: return '1'
    
    l =  int(log2(x)) 
    a = x - 2**(int(log2(x)))
    k = int(log2(x))
    
    return unary(l) + binary(a,k)

def elias_delta(x):
    if x == 0: return '0'
    if x == 1: return '1'
     
#    l =  1 + int(log2(x)) 
    a = x - 2**(int(log2(x)))   # reminder
    k = int(log2(x))            # division
    
    x = int(log2(x))+1
    return elias_gamma(x) + binary(a,k)
	
def golomb(b, x):
	q = int((x) / b)
	r = int((x) % b)
	
	l = int(ceil(log2(b)))
	#print q,r,l

	return unary(q) + binary(r, l)

def golomb8(data):
    return golomb(8,data)
    
def golomb4(data):
    return golomb(4,data)
    
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

				

