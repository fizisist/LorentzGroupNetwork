# dot.py
# Author: Jan Offermann
# Date: 03/19/20

# Works for N+1 dim. Minkowski space, w/ metric signature (+,-,-,...,-).
import numpy as np
from numba import jit

# Given two 4-momenta, return the dot product.
@jit
def dot(p1,p2):
    return p1[0] * p2[0] - np.dot(p1[1:],p2[1:])

# Given two (N,1) lists of 4-momenta, return list of (N) dot products.
@jit
def dots(p1s,p2s):
    return np.array([dot(p1s[i],p2s[i]) for i in range(p1s.shape[0])])

# Given one (N,1) list of 4-momenta, return list of (N) norms.
@jit
def masses(p1):
    return np.sqrt(np.maximum(0.,dots(p1,p1)))

# Given two (N,1) lists of 4-momenta, return array (N,N) of dot products.
# Here, position (i,j) is the dot product of the i'th element of the first
# list and the j'th element of the second list.
@jit
def dots_matrix_multi(p1s,p2s):
    a = p1s.shape[0]
    b = p2s.shape[0]
    return np.array([[dot(p1s[i],p2s[j]) for j in range(b)] for i in range(a)])

# Given one (N,1) lists of 4-momenta, return array (N,N) of dot products.
# Here, position (i,j) is the dot product of the i'th element of the list
# and the j'th element of the list.
@jit
def dots_matrix_single(p1s):
    a = p1s.shape[0]
    matrix = np.zeros((a,a),dtype=np.dtype('f8'))
    for i in range(a):
        for j in range(i+1): # get the diagonal elements
            matrix[i,j] = dot(p1s[i],p1s[j])
            matrix[j,i] = matrix[i,j] # symmetric
    return matrix
    
# Handler for dots_matrix_single & dots_matrix_multi.
# The former is much faster, and should be used if
# p1s == p2s. It may still be faster to directly call
# the underlying functions, since it avoids this
# (somewhat costly?) check on array equality.
def dots_matrix(p1s,p2s):
    if(np.array_equal(p1s,p2s)):
        return dots_matrix_single(p1s)
    else:
        return dots_matrix_multi(p1s,p2s)

def test(N = 1000):
    import time
    p1s = np.random.rand(N,8,4)
    p2s = np.random.rand(N,8,4)
    start = time.time()
    for i in range(N):
        a = dots_matrix(p1s[i],p2s[i])
    end = time.time()
    print('t1 = ', end-start)
    start = time.time()
    for i in range(N):
        a = dots_matrix(p1s[i],p1s[i])
    end = time.time()
    print('t2 = ', end-start)
    return
