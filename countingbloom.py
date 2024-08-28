import math
from fnvhash import fnv1a_32
from bitarray import bitarray
from bitarray.util import ba2int,int2ba
import mmh3
import sys

# Reference: https://github.com/yankun1992/fastbloom/blob/main/fastbloom-rs/src/builder.rs

if sys.maxsize == 2**31 - 1:  # 32-bit platform
    SUFFIX = 0b0001_1111
    MASK = 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11100000
else:  # 64-bit platform
    SUFFIX = 0b0011_1111
    MASK = 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11000000

def optimal_m(n, p):
    fact = -(n * math.log(p))
    div = (2 * math.log(2)) ** 2
    m = fact / div
    m = math.ceil(m)

    if (m & SUFFIX) != 0:
        m = (m & MASK) + SUFFIX + 1

    return int(m)

def optimal_k(n, m):
    k = (m * math.log(2)) / n
    k = math.ceil(k)
    return int(k)

class CBloomFilter():
    def __init__(self, n, counter_size, p=0.01):
        
        self.n=n
        self.N=counter_size
        self.m=optimal_m(n, p)
        self.k=10#optimal_k(n, self.m)

        self.bit_array = []
        for i in range(self.m):
            count=bitarray(self.N)
            count.setall(0)
            self.bit_array.append(count)

    def hash(self,item,seed):
        return mmh3.hash(item.encode(),seed) % self.m

    def add(self, item):

        for i in range(self.k):
            index = self.hash(item,i)

            cur_val=ba2int(self.bit_array[index])
            new_array=int2ba(cur_val+1,length=self.N)
            
            self.bit_array[index]=new_array
            
    def check(self, item):
        for i in range(self.k):
            index = self.hash(item,i)
            cur_val=ba2int(self.bit_array[index])

            if(not cur_val>0):
                return False
        return True
    
    def remove(self,item):
        if(self.check(item)):
            for i in range(self.k):
                index = self.hash(item,i)
                
                cur_val=ba2int(self.bit_array[index])
                new_array=int2ba(cur_val-1,length=self.N)
                self.bit_array[index]=new_array

            print('Element Removed')
        else:
            print('Element is probably not exist')
    
    def get_count(self, code):
        cur_val = 1e15
        for i in range(self.k):
            index = self.hash(code,i)
            cur_val = min(cur_val, ba2int(self.bit_array[index]))
        return cur_val
