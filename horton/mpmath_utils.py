# coding: UTF-8
"""Functions for efficient and convenient handling of
NumPy arrays holding mpmath numbers
"""
import numpy as np
import mpmath as mp
import copy

# Operations that do work on NumPy arrays of mpmath objects:
# +, -, *scalar
# **2
# abs()
# conj()
# np.dot()
#
# ...

# Operations that do not work on NumPy arrays of mpmath objects:
# .conj()
# .real
# .imag
#
# ...

def array2mp(a):
    """convert a NumPy array of float or complex
    to a NumPy array of mpmath objects (mpf/mpc)"""
    if a.dtype == np.complex:
        func = _cfunc_mpc
    else:
        func = _cfunc_mpf
    tmp = a.ravel()
    res = func(tmp)
    return res.reshape(a.shape)

def mparray2npfloat(a):
    """convert a NumPy array of mpmath objects (mpf/mpc)
    to a NumPy array of float"""
    tmp = a.ravel()
    res = _cfunc_float(tmp)
    return res.reshape(a.shape)

def mparray2npcomplex(a):
    """convert a NumPy array of mpmath objects (mpf/mpc)
    to a NumPy array of complex"""
    tmp = a.ravel()
    res = _cfunc_complex(tmp)
    return res.reshape(a.shape)

def mpfzeros(shape):
    """construct an NumPy array of shape "shape"
    filled with zero mpf's
    """
    res = np.empty(shape, object)
    res2 = res.ravel()  # this is a view on res!
    for i in xrange(len(res2)):
        # it would be faster to just copy this object
        # rather than initialize it again and again
        res2[i] = mp.mpf(0.0)
        # this is sloooow
        # res2[i] = copy.deepcopy(_mpf0)
        # see also http://code.activestate.com/recipes/66507/
    return res

def mpczeros(shape):
    """construct an NumPy array of shape "shape"
    filled with zero mpc's
    """
    res = np.empty(shape, object)
    res2 = res.ravel()  # this is a view on res!
    for i in xrange(len(res2)):
        # it would be faster to just copy this object
        # rather than initialize it again and again
        res2[i] = mp.mpc(0.0)
        # this is sloooow
        # res2[i] = copy.deepcopy(_mpc0)
        # see also http://code.activestate.com/recipes/66507/
    return res

def apply2mparray(a, func, vfunc = None):
    """apply a function to a NumPy array of mpmath objects
    Parameters:
        func: function to apply. will be vectorize()d. ignored if vfunc != None
       vfunc: a vectorized function. may be used instead
              of func in order to avoid the overhead of vectorize()
    """
    a2 = a.ravel()
    if vfunc is None:
        vfunc = np.vectorize(func, otypes=(_dtype_object,))
    res = vfunc(a2)
    return res.reshape(a.shape)


_mpf0 = mp.mpf(0.0)
_mpc0 = mp.mpc(0.0)
_dtype_object = np.dtype('object')
_cfunc_mpc = np.vectorize(mp.mpc, otypes=(_dtype_object,))
_cfunc_mpf = np.vectorize(mp.mpf, otypes=(_dtype_object,))
_cfunc_complex = np.vectorize(complex)
_cfunc_float   = np.vectorize(float)
