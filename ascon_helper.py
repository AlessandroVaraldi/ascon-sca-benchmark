import sys, logging
import numpy as np
from itertools import chain
import random

reg_size = 64
ASCON_128_IV_int = 0x80400c0600000000
ASCON_128_IV = (0x80400c0600000000).to_bytes(16, 'big')

ASCON_SBOX_NP = np.array([0x04, 0x0b, 0x1f, 0x14, 0x1a, 0x15, 0x09, 0x02, 
              0x1b, 0x05, 0x08, 0x12, 0x1d, 0x03, 0x06, 0x1c, 
              0x1e, 0x13, 0x07, 0x0e, 0x00, 0x0d, 0x11, 0x18, 
              0x10, 0x0c, 0x01, 0x19, 0x16, 0x0a, 0x0f, 0x17])
ASCON_SBOX_NP_DIM = [np.array([ASCON_SBOX_NP[x] for x in range(1<<4)]), np.array([ASCON_SBOX_NP[x] for x in range(1<<4, 1<<5)])]#If first bit is 0 or 1

#diffusion layer on register x_i
dr = [[45,36],
      [3,25],
      [63,58],
      [54,47],
      [57,23]]

def bit_value(i, register_value):
    return (register_value>>(reg_size-1-i))&1
    
def select_bit_bytes(i, array):
    """Returns the bit index `i` from the array of bytes `array`
    If array have 2 dimensions, it will select bits given the first axis.
    
    :param i: index of the bit to select.
    :type i: int
    :param aray: array of bytes in bytes/int format.
    :type array: list
    """
    if isinstance(array,np.ndarray):
        if len(array.shape)==1:
            i = i % (len(array)*8)
            return (array[i//8]>>(7-(i%8)))&1
        elif len(array.shape)==2:
            i = i % (len(array[0])*8)
            return (array[:,i//8]>>(7-(i%8)))&1
    elif isinstance(array, bytes) or isinstance(array, list):
        i = i % (len(array)*8)
        return (array[i//8]>>(7-(i%8)))&1
    else: 
        logging.error("select_bit_bytes: array type not supported, got {}".format(type(array)))
        logging.exception('')

#-----------------------------------------#
# Intermediate value of SBox and Linear Diffusion Layer
reg_num_0 = 4
reg_num_1 = 1
def intermediate_value(keyguess, trace_parameters):
    """Return the intermediate value of a key in the SBox . 
    nonce0 and nonce1 should be byte arrays of size (8) as they both are half of the nonce length (16 bytes).
    k should be a 3-bit value as an integer. This bit corresponds to the part of the key concerned in the attack given subkey.

    :param trace_parameters: Dictionary containing the trace parameters for intermediate value computation
    :type trace_parameters: dict
    :return: Intermediate value
    :rtype: Any
    """        
    if trace_parameters['subkey']<reg_size:
        y0_i = lambda x0,x1,x3,x4: x4&(x1^1)^x3
        res = (y0_i(select_bit_bytes(trace_parameters['subkey']%reg_size                   ,ASCON_128_IV),   (keyguess>>2)&1, select_bit_bytes(trace_parameters['subkey']%reg_size                   , trace_parameters['nonce0']), select_bit_bytes(trace_parameters['subkey']%reg_size           , trace_parameters['nonce1']))) \
            ^ (y0_i(select_bit_bytes((trace_parameters['subkey']+dr[reg_num_0][0])%reg_size,ASCON_128_IV),   (keyguess>>1)&1, select_bit_bytes((trace_parameters['subkey']+dr[reg_num_0][0])%reg_size, trace_parameters['nonce0']), select_bit_bytes((trace_parameters['subkey']+dr[4][0])%reg_size, trace_parameters['nonce1']))) \
            ^ (y0_i(select_bit_bytes((trace_parameters['subkey']+dr[reg_num_0][1])%reg_size,ASCON_128_IV),   (keyguess>>0)&1, select_bit_bytes((trace_parameters['subkey']+dr[reg_num_0][1])%reg_size, trace_parameters['nonce0']), select_bit_bytes((trace_parameters['subkey']+dr[4][1])%reg_size, trace_parameters['nonce1'])))
        return res
    elif trace_parameters['subkey']<2*reg_size:
        y0_i = lambda x0,x12,x3,x4: x3&(x12^1)^x4
        res = (y0_i(select_bit_bytes(trace_parameters['subkey']%reg_size                   ,ASCON_128_IV),(keyguess>>2)&1, select_bit_bytes(trace_parameters['subkey']%reg_size                   , trace_parameters['nonce0']), select_bit_bytes(trace_parameters['subkey']%reg_size                   , trace_parameters['nonce1']))) \
            ^ (y0_i(select_bit_bytes((trace_parameters['subkey']+dr[reg_num_0][0])%reg_size,ASCON_128_IV),(keyguess>>1)&1, select_bit_bytes((trace_parameters['subkey']+dr[reg_num_1][0])%reg_size, trace_parameters['nonce0']), select_bit_bytes((trace_parameters['subkey']+dr[reg_num_1][0])%reg_size, trace_parameters['nonce1']))) \
            ^ (y0_i(select_bit_bytes((trace_parameters['subkey']+dr[reg_num_0][1])%reg_size,ASCON_128_IV),(keyguess>>0)&1, select_bit_bytes((trace_parameters['subkey']+dr[reg_num_1][1])%reg_size, trace_parameters['nonce0']), select_bit_bytes((trace_parameters['subkey']+dr[reg_num_1][1])%reg_size, trace_parameters['nonce1'])))
        return res
    
def select_subkey(i, key, array=False):
    if i<reg_size:
        sub_key = [select_bit_bytes((i)%reg_size         , key[0:8]), 
                   select_bit_bytes((i+dr[reg_num_0][0])%reg_size, key[0:8]), 
                   select_bit_bytes((i+dr[reg_num_0][1])%reg_size, key[0:8])]
    elif i<2*reg_size:
        sub_key = [x^y for x,y in zip([select_bit_bytes((i)%reg_size         , key[0:8]), 
                                       select_bit_bytes((i+dr[reg_num_1][0])%reg_size, key[0:8]), 
                                       select_bit_bytes((i+dr[reg_num_1][1])%reg_size, key[0:8])], 
                                      [select_bit_bytes((i)%reg_size         , key[8:16]), 
                                       select_bit_bytes((i+dr[reg_num_1][0])%reg_size, key[8:16]), 
                                       select_bit_bytes((i+dr[reg_num_1][1])%reg_size, key[8:16])]
                                    )
        ]
    if not array:
        sub_key = sub_key[0]<<2 | sub_key[1]<<1 | sub_key[2]
    return sub_key
#-----------------------------------------#


#-----------------------------------------#
# Software protected utilities

ASCON_EXTERN_BI = 0
ASCON_ROR_SHARES = 5

def bswap64(x):
    y = np.zeros((len(x)+7)//8*8, dtype=np.uint8)
    for i in range(len(x)):
        y[i // 8 * 8 + 7 - i % 8] = x[i]
    return y

def ror(v, r, s):
    return ((v & (2**s - 1)) >> (r % s) | (v << (s - (r % s)) & (2**s - 1)))

def tobi(x):
    y = np.zeros((len(x)+7)//8*8, dtype=np.uint8)
    for i in range(len(x)):
        for j in range(4):
            e = (x[i] >> (j * 2)) & 1
            o = (x[i] >> (j * 2 + 1)) & 1
            y[(i//8)*8+(i%8)//2] |= e << ((i%2) * 4 + j)
            y[(i//8)*8+(i%8)//2+4] |= o << ((i%2) * 4 + j)
    return y

def frombi(x):
    y = np.zeros((len(x)+7)//8*8, dtype=np.uint8)
    for i in range(len(x)):
        for j in range(4):
            e = (x[(i//8)*8+(i%8)//2] >> ((i%2) * 4 + j)) & 1
            o = (x[(i//8)*8+(i%8)//2+4] >> ((i%2) * 4 + j)) & 1
            y[i] |= e << (j * 2)
            y[i] |= o << (j * 2 + 1)
    return y

def size_shares(l, ns):
    return ((l + 7) // 8) * 8 * ns

def cw_generate_shares(x, ns):
    l = size_shares(len(x), ns) // ns
    xs = np.zeros((l // 4, ns, 4), dtype=np.uint8)
    x = bswap64(x)
    if ASCON_EXTERN_BI:
        x = tobi(x)
    for i in range(len(x)):
        xs[i // 4][0][i % 4] = x[i]
    for i in range(0, l // 4, 2):
        for j in range(1, ns):
            r0 = random.getrandbits(32)
            r1 = random.getrandbits(32)
            if ASCON_EXTERN_BI:
                m = ror(r1, ASCON_ROR_SHARES * j, 32) << 32 | ror(r0, ASCON_ROR_SHARES * j, 32)
            else:
                m = ror(r1 << 32 | r0, 2 * ASCON_ROR_SHARES * j, 64)
            for k in range(4):
                xs[i + 0][0][k] ^= (m >> (k * 8)) & 0xff
                xs[i + 1][0][k] ^= (m >> (k * 8 + 32)) & 0xff
                xs[i + 0][j][k] ^= (r0 >> (k * 8)) & 0xff
                xs[i + 1][j][k] ^= (r1 >> (k * 8)) & 0xff
    xs = xs.flatten()
    return bytearray(xs)

def cw_combine_shares(xs, ns):
    l = len(xs) // ns
    xs = np.reshape(xs, (l // 4, ns, 4))
    x = np.zeros((l // 4, 4), dtype=np.uint8)
    for i in range(0, l // 4, 2):
        for j in range(ns):
            r0 = r1 = 0
            for k in range(4):
                r0 |= int(xs[i + 0][j][k]) << (k * 8)
                r1 |= int(xs[i + 1][j][k]) << (k * 8)
            if ASCON_EXTERN_BI:
                m = ror(r1, ASCON_ROR_SHARES * j, 32) << 32 | ror(r0, ASCON_ROR_SHARES * j, 32)
            else:
                m = ror(r1 << 32 | r0, 2 * ASCON_ROR_SHARES * j, 64)
            for k in range(4):
                x[i + 0][k] ^= (m >> (k * 8)) & 0xff
                x[i + 1][k] ^= (m >> (k * 8 + 32)) & 0xff
    x = x.flatten()
    if ASCON_EXTERN_BI:
        x = frombi(x)
    x = bswap64(x)
    return x

#-----------------------------------------#
# Hardware protected utilities

def hw_generate_shares(x, ns):
    randomness = np.random.randint(0,256,(ns-1,len(x)), dtype=np.uint8)
    if ns==2:
        shares = np.vstack(([x^y for x,y in zip(x, randomness[0])], [x for x in randomness[0]])).reshape((ns,4,-1))
    elif ns>2:
        shares = np.vstack(([x^y for x,y in zip(x, randomness[0])],[[x^y for x,y in zip(randomness[i], randomness[i+1])] for i in range(ns-2)], [x for x in randomness[-1]])).reshape((ns,4,-1))
    return b''.join(np.array([[shares[j][i] for j in range(ns)] for i in range(0,len(x)//4)], dtype=np.uint8).flatten())

def hw_combine_shares(x, ns):
    return sum([[sum(shares)&0xFF for shares in zip(*[x[i+j*4:i+4*(j+1)] for j in range(ns)])] for i in range(0,len(x),4*ns)], [])
