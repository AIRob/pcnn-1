#encoding:UTF-8

import os
import math
import numpy as np

def get_primes():
    D = {}
    q = 2
    while True:
        if q not in D:
            yield q
            D[q*q] = [q]
        else:
            for p in D[q]:
                D.setdefault(p+q, []).append(p)
            del D[q]
        q += 1
        
        
def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))
        

if __name__ == '__main__':
    a = get_primes()
    for _ in range(227):
        b = next(a)
        print(b)
        print(is_prime(b))