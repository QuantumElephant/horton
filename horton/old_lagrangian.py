import numpy as np
import copy as cp
import scipy.optimize as op
import scipy.linalg.matfuncs as mat
from horton.newton import reallyStupidNewton as newt 

#TODO: profile to figure out a quick way of evaluating this function.
class old_lagrangian(object):
    def __init__(self, T, V, S, N, dVee, W, debug = False):
        """Takes fixed integrals and functionals.
            The integrals need to be instances of the LinearAlgebraFactory.
            N is an array containing individual integrals for alpha and beta spins.

          - T is the kinetic integral
          - V is an array of nuclear-electron integrals (alpha/beta spin)
          - S is the overlap integral
          - N is an array of the (scalar) normalization values for the density matrix ( alpha and beta spin )

          - dVee is an instance of an object for the correction to the electron-electron interaction functional.
              It includes three methods, dD, dDD, dDaDb for the derivatives with respect to the density matrix.
        """
        self.debug = debug
        self.zero_pauli = False
        self.T = T
        self.V = V
        self.S = S
        self.N = N
        self.W = W

        self.dVee = dVee
        
        self.nbasis = self.S.nbasis
        self.inputShape = S._array.shape
        self.fine_tune = False
        
        self.DSize = None
        self.grad_off = None
        self.grad_size = None
        
        self.recalc_offsets()

#    def gradToVec(self, x):
#        view = []
#        for i in x:
#            for j in i:
#                view.append(j._array.ravel())
#        view = np.hstack(view)
#        return view
        
#    def gradToMat(self, v, mat):
#        result = [[np.zeros_like(mat[0][0]._array), np.zeros_like(mat[0][1]._array)]
#                  [np.zeros_like(mat[1][0]._array), np.zeros_like(mat[1][1]._array)]
#                  [np.zeros_like(mat[2][0]._array), np.zeros_like(mat[2][1]._array)]]
#                  
#        x_off = self.grad_off
#        
#        result[0][0] = v[x_off[0]:x_off[1]].reshape(self.inputShape)
#        result[0][1] = v[x_off[1]:x_off[2]].reshape(self.inputShape)
#        result[1][0] = v[x_off[2]:x_off[3]].reshape(self.inputShape)
#        result[1][1] = v[x_off[3]:x_off[4]].reshape(self.inputShape)
#        result[2][0] = v[x_off[4]]
#        result[2][1] = v[x_off[5]]
#        
#        return result
    
    def recalc_offsets(self):
        self.DSize = self.nbasis**2
        self.grad_size = [0, self.DSize,self.DSize, self.DSize, self.DSize, 1, 1]
        self.grad_off = np.cumsum(self.grad_size)

    def rebuild(self,dx,x,lin_x = None, mask = None): #TODO: generalize to arbitrary number of X
        """This function takes an ndarray dx, along with the old input parameters x (which are one-body instances).
        This updates x with the new values of dx
        """
        damp_factor = 1
#        damp_factor = 10e-7; print("warning, using damp factor")
        dx *= damp_factor

        if mask is not None:
            dx *= mask

        if lin_x is not None:
            lin_x += dx 

        x_off = self.grad_off

        x[0][0]._array += dx[x_off[0]:x_off[1]].reshape(self.inputShape)
        x[0][1]._array += dx[x_off[1]:x_off[2]].reshape(self.inputShape)
        x[1][0]._array += dx[x_off[2]:x_off[3]].reshape(self.inputShape)
        x[1][1]._array += dx[x_off[3]:x_off[4]].reshape(self.inputShape)
        x[2][0]._array += dx[x_off[4]]
        x[2][1]._array += dx[x_off[5]]
        
        if lin_x is None:
            return x
         
        return x, lin_x

    def energy(self, D):
        t = self.T._array

        result = 0.
        result += self.dVee.Vee(D)
        
        for i in [0,1]:
            dm = D[i]._array
            v = self.V[i]._array
            result += np.trace(np.dot((t-v),dm))
            
        return result

    def dummy_lagrangian(self, D, B, Mu):
        results = 0
        for i in [0,1]:
            results += np.sum(D[i]._array)**2 + np.sum(B[i]._array)**2 + Mu[i]._array**2

        return results

    def lagrangian(self, D, B, Mu):
        s = self.S._array

        for i in [0,1]:
            D[i]._array = 0.5*(D[i]._array+D[i]._array.T)
            B[i]._array = 0.5*(B[i]._array+B[i]._array.T)

#        result = 0
        result = self.energy(D)

        for i in [0,1]:
            dm = D[i]._array
            mu = Mu[i]._array
            n = self.N[i]

            #result += self.W #not implemented yet
            pauli_test = self._pauli_constraint(D[i], B[i])
            result -= pauli_test
            result -= mu*(np.trace(np.dot(dm,s)) - n)
        return result
    
    def new_lagrangian(self, D, B, Mu):
        s = self.S._array

        for i in [0,1]:
            D[i]._array = 0.5*(D[i]._array+D[i]._array.T)
            B[i]._array = 0.5*(B[i]._array+B[i]._array.T)

#        result = 0
        result = self.energy(D)

        for i in [0,1]:
            dm = D[i]._array
            mu = Mu[i]._array
            n = self.N[i]

            #result += self.W #not implemented yet
            pauli_test = self._new_pauli_constraint(D[i], B[i])
            result -= pauli_test
            result -= mu*(np.trace(np.dot(dm,s)) - n)
        return result

#    def ugly_lagrangian(self, dma, dmb, ba, bb, mua, mub):
#        t = self.T._array
#        s = self.S._array
#
#        D = [dma, dmb]
#        B = [ba,bb]
#        Mu = [mua,mub]
#
#        result = 0.
#        result += self.dVee.Vee(D)
#
#        for i in [0,1]:
#            D[i]._array = 0.5*(D[i]._array + D[i]._array.T)
#            B[i]._array = 0.5*(B[i]._array + B[i]._array.T)
#            
#            dm = D[i]._array
#            b = B[i]._array
#            mu = Mu[i]
#            n = self.N[i]
#            v = self.V[i]._array
#
#            result += np.trace(np.dot((t-v),dm))
#
#            #result += self.W #not implemented yet
#
#            if not self.zero_pauli:
#                pauli_test = self._pauli_constraint(D[i], B[i])
##                if pauli_test < 0 or self.debug:
#                #print("lagrangian: pauli_tripped")
#                result -= pauli_test
#            result -= mu._array*(np.trace(np.dot(dm,s)) - n)
#        return result

    def fdiff_dummy_gradient(self, *args):
        fn = self.dummy_lagrangian
        
        return self.fdiff_ordered_sym_gradient(fn, *args)    


    def fdiff_ordered_lambda_gradient(self, *args):
        fn = self.lagrangian
        
        return self.fdiff_ordered_sym_gradient(fn, *args)    
    
    def fdiff_new_idem_gradient(self, *args):
        fn = self.new_lagrangian
        
        return self.fdiff_ordered_sym_gradient(fn, *args) 


    def fdiff_ordered_sym_gradient(self, fn, *args):
        h = 1e-3
    
        result = []
    
        tmpFwdArgs = cp.deepcopy(args)
        tmpBackArgs = cp.deepcopy(args)
    
        tmpFwd = list(tmpFwdArgs)
        tmpBack = list(tmpBackArgs)
    
        for iKey,i in enumerate(args): # iterating over D, B, Mu
            for kKey,k in enumerate(i):
    #            print(iKey, kKey)
    
                intermediate_result = np.zeros(k._array.size)
    
                it = np.nditer(intermediate_result, flags=['multi_index'])
                while not it.finished:
                    if k._array.size == 1:
                        dim2_index = 0
                    else:
                        dim2_index = np.unravel_index(it.multi_index[0], k._array.shape)
    
                    tmpFwd[iKey][kKey]._array[dim2_index] += h
                    tmpBack[iKey][kKey]._array[dim2_index] -= h
    
                    intermediate_result[it.multi_index] = (fn(*tmpFwd) - fn(*tmpBack)) / (2*h)
    
                    tmpFwd[iKey][kKey]._array[dim2_index] -= h
                    tmpBack[iKey][kKey]._array[dim2_index] += h
    
                    it.iternext()
        #            print(intermediate_result)
#                intermediate_result = 0.5*(intermediate_result + np.swapaxes(intermediate_result, 0, 1))
                result.append(intermediate_result)
        #            print(inner_result)
        result = np.hstack(result)
    
    #    print(result)
        return result

#    def fdiff_sym_gradient(self, *args):
#        h = 10e-4
#    
#        d0 = args[0][0]
#        d1 = args[0][1]
#        b0 = args[1][0]
#        b1 = args[1][1]
#    
#    #    fn = stupid_fn
#        fn = self.lagrangian
#    
#        result = []
#    
#        if self._pauli_constraint(d0, b0) > 0 or self._pauli_constraint(d1, b1) > 0:
#            self.zero_pauli = True
#        else:
#            self.zero_pauli = False
#    
#        tmpFwdArgs = cp.deepcopy(args)
#        tmpBackArgs = cp.deepcopy(args)
#    
#        tmpFwd = list(tmpFwdArgs)
#        tmpBack = list(tmpBackArgs)
#    
#        for iKey,i in enumerate(args): # iterating over D, B, Mu
#            for kKey,k in enumerate(i):
#    #            print(iKey, kKey)
#    
#                intermediate_result = np.zeros(k._array.size)
#    
#                it = np.nditer(intermediate_result, flags=['multi_index'])
#                while not it.finished:
#                    if k._array.size == 1:
#                        dim2_index = 0
#                    else:
#                        dim2_index = np.unravel_index(it.multi_index[0], k._array.shape)
#    
#                    tmpFwd[iKey][kKey]._array[dim2_index] += h
#                    tmpBack[iKey][kKey]._array[dim2_index] -= h
#    
#                    intermediate_result[it.multi_index] = (fn(*tmpFwd) - fn(*tmpBack)) / (2*h)
#    
#                    tmpFwd[iKey][kKey]._array[dim2_index] -= h
#                    tmpBack[iKey][kKey]._array[dim2_index] += h
#    
#                    it.iternext()
#        #            print(intermediate_result)
##                intermediate_result = 0.5*(intermediate_result + np.swapaxes(intermediate_result, 0, 1))
#                result.append(intermediate_result)
#        #            print(inner_result)
#        result = np.hstack(result)
#    
#    #    print(result)
#        return result
        
#    def fdiff_sym_hessian(self, *compressed_args):
#        h = 10e-4
#        
#        args = []
#        
##        for i in compressed_args:
##            for j in i:
##                args = args.append(j)
#        
#        d0 = compressed_args[0][0]
#        d1 = compressed_args[0][1]
#        b0 = compressed_args[1][0]
#        b1 = compressed_args[1][1]
#        mu0 = compressed_args[2][0]
#        mu1 = compressed_args[2][1]
#        
#        args = [d0, d1, b0, b1, mu0, mu1]
#        result = []
#    
#    #    fn = stupid_fn
#        fn = self.ugly_lagrangian
#        
#        if self._pauli_constraint(d0, b0) > 0 or self._pauli_constraint(d1, b1) > 0: #TODO: split this up
#            self.zero_pauli = True
#            
#        else:
#            self.zero_pauli = False
#        
#        tmpFwdArgs = cp.deepcopy(args)
#        tmpBackArgs = cp.deepcopy(args)
#        tmpFwdBackArgs = cp.deepcopy(args)
#        tmpBkFwdArgs = cp.deepcopy(args)
#    
#        tmpFwd = list(tmpFwdArgs)
#        tmpBack = list(tmpBackArgs)
#        tmpFwdBack = list(tmpFwdBackArgs)
#        tmpBkFwd = list(tmpBkFwdArgs)
#    
#        for iKey,i in enumerate(args): # iterating over D, B, Mu
#            inner_result = []
#            for kKey,k in enumerate(args): # iterating over D, B, Mu again
#                print(iKey, kKey)
#    
#                intermediate_result = np.zeros([i._array.size, k._array.size])
#    
#                it = np.nditer(intermediate_result, flags=['multi_index'])
#                while not it.finished:
#    #                dim1_index = np.unravel_index(it.multi_index[0], k._array.shape)
#    #                dim2_index = np.unravel_index(it.multi_index[1], i._array.shape)
#    
#                    if k._array.size == 1 and i._array.size == 1:
#                        dim1_index = 0
#                        dim2_index = 0
#                    elif k._array.size == 1:
#                        dim1_index = 0
#                        dim2_index = np.unravel_index(it.multi_index[0], i._array.shape)
#                    elif i._array.size == 1:
#                        dim1_index = np.unravel_index(it.multi_index[1], k._array.shape)
#                        dim2_index = 0
#                    else:
#                        dim1_index = np.unravel_index(it.multi_index[0], k._array.shape)
#                        dim2_index = np.unravel_index(it.multi_index[1], i._array.shape)
#    
#                    tmpFwd[kKey]._array[dim1_index] += h
#                    tmpFwd[iKey]._array[dim2_index] += h
#    
#                    tmpBack[kKey]._array[dim1_index] -= h
#                    tmpBack[iKey]._array[dim2_index] -= h
#    
#                    tmpFwdBack[kKey]._array[dim1_index] += h
#                    tmpFwdBack[iKey]._array[dim2_index] -= h
#    
#                    tmpBkFwd[kKey]._array[dim1_index] -= h
#                    tmpBkFwd[iKey]._array[dim2_index] += h
#    
#                    intermediate_result[it.multi_index] = (fn(*tmpFwd) - fn(*tmpFwdBack) - fn(*tmpBkFwd) + fn(*tmpBack)) / (4*(h**2))
#    
#                    tmpFwd[kKey]._array[dim1_index] -= h
#                    tmpFwd[iKey]._array[dim2_index] -= h
#    
#                    tmpBack[kKey]._array[dim1_index] += h
#                    tmpBack[iKey]._array[dim2_index] += h
#    
#                    tmpFwdBack[kKey]._array[dim1_index] -= h
#                    tmpFwdBack[iKey]._array[dim2_index] += h
#    
#                    tmpBkFwd[kKey]._array[dim1_index] += h
#                    tmpBkFwd[iKey]._array[dim2_index] -= h
#    
#                    #print(intermediate_result)
#                    #print(it.multi_index)
#    
#                    it.iternext()
#    #            print(intermediate_result)
##                intermediate_result = 0.5*(intermediate_result + np.swapaxes(intermediate_result, 0, 1))
#                inner_result.append(intermediate_result)
#    #            print(inner_result)
#            result.append(np.hstack(inner_result))
#            
#        result = np.vstack(result)
#    
#    #    print(result)
#        return result
#
    def fdiff_dummy_hessian(self, *args):
        fn = self.dummy_lagrangian
        
        return self.fdiff_ordered_sym_hessian(fn, *args)

    def fdiff_ordered_lambda_hessian(self, *args):
        fn = self.lagrangian
        
        return self.fdiff_ordered_sym_hessian(fn, *args)
    
    def fdiff_new_idem_hessian(self, *args):
        fn = self.new_lagrangian
        
        return self.fdiff_ordered_sym_hessian(fn, *args)

    def fdiff_ordered_sym_hessian(self,fn, *args):
        h = 1e-3
        result = []
        
        tmpFwdArgs = cp.deepcopy(args)
        tmpBackArgs = cp.deepcopy(args)
        tmpFwdBackArgs = cp.deepcopy(args)
        tmpBkFwdArgs = cp.deepcopy(args)
    
        tmpFwd = list(tmpFwdArgs)
        tmpBack = list(tmpBackArgs)
        tmpFwdBack = list(tmpFwdBackArgs)
        tmpBkFwd = list(tmpBkFwdArgs)
    
        for iKey,i in enumerate(args): # iterating over D, B, Mu
            for jKey, j in enumerate(i):
                inner_result = []
                for kKey,k in enumerate(args): # iterating over D, B, Mu again
                    for lKey, l in enumerate(k):
                        
#                        print(iKey, kKey)
                        intermediate_result = np.zeros([j._array.size, l._array.size])
                        it = np.nditer(intermediate_result, flags=['multi_index'])
                        while not it.finished:
                            if j._array.size == 1 and l._array.size == 1:
                                dim1_index = 0
                                dim2_index = 0
                            elif l._array.size == 1:
                                dim1_index = 0
                                dim2_index = np.unravel_index(it.multi_index[0], j._array.shape)
                            elif j._array.size == 1:
                                dim1_index = np.unravel_index(it.multi_index[1], l._array.shape)
                                dim2_index = 0
                            else:
                                dim1_index = np.unravel_index(it.multi_index[0], l._array.shape)
                                dim2_index = np.unravel_index(it.multi_index[1], j._array.shape)
                            
                            tmpFwd[kKey][lKey]._array[dim1_index] += h
                            tmpFwd[iKey][jKey]._array[dim2_index] += h
                            
                            tmpBack[kKey][lKey]._array[dim1_index] -= h
                            tmpBack[iKey][jKey]._array[dim2_index] -= h
                            
                            tmpFwdBack[kKey][lKey]._array[dim1_index] += h
                            tmpFwdBack[iKey][jKey]._array[dim2_index] -= h
                            
                            tmpBkFwd[kKey][lKey]._array[dim1_index] -= h
                            tmpBkFwd[iKey][jKey]._array[dim2_index] += h
                            
                            intermediate_result[it.multi_index] = (fn(*tmpFwd) - fn(*tmpFwdBack) - fn(*tmpBkFwd) + fn(*tmpBack)) / (4*(h**2))
                            
                            tmpFwd[kKey][lKey]._array[dim1_index] -= h
                            tmpFwd[iKey][jKey]._array[dim2_index] -= h
                            
                            tmpBack[kKey][lKey]._array[dim1_index] += h
                            tmpBack[iKey][jKey]._array[dim2_index] += h
                            
                            tmpFwdBack[kKey][lKey]._array[dim1_index] -= h
                            tmpFwdBack[iKey][jKey]._array[dim2_index] += h
                            
                            tmpBkFwd[kKey][lKey]._array[dim1_index] += h
                            tmpBkFwd[iKey][jKey]._array[dim2_index] -= h
                            
                            #print(intermediate_result)
                            #print(it.multi_index)
                            
                            it.iternext()
                        #            print(intermediate_result)
                        #                intermediate_result = 0.5*(intermediate_result + np.swapaxes(intermediate_result, 0, 1))
                        inner_result.append(intermediate_result)
                    #            print(inner_result)
                result.append(np.hstack(inner_result))
            
        result = np.vstack(result)
    
    #    print(result)
        return result
    
#    def fdiff_slow_sym_hessian(self,fn, *args):
#        h = 1e-3
#        result = []
#        
#        tmpFwdArgs = cp.deepcopy(args)
#        tmpBackArgs = cp.deepcopy(args)
#        tmpFwdBackArgs = cp.deepcopy(args)
#        tmpBkFwdArgs = cp.deepcopy(args)
#    
#        tmpFwd = list(tmpFwdArgs)
#        tmpBack = list(tmpBackArgs)
#        tmpFwdBack = list(tmpFwdBackArgs)
#        tmpBkFwd = list(tmpBkFwdArgs)
#    
#        for iKey,i in enumerate(args): # iterating over D, B, Mu
#            for jKey, j in enumerate(i):
#                inner_result = []
#                for kKey,k in enumerate(args): # iterating over D, B, Mu again
#                    for lKey, l in enumerate(k):
#                        
##                        print(iKey, kKey)
#                        intermediate_result = np.zeros([j._array.size, l._array.size])
#                        it = np.nditer(intermediate_result, flags=['multi_index'])
#                        while not it.finished:
#                            tmpFwdArgs = cp.deepcopy(args)
#                            tmpBackArgs = cp.deepcopy(args)
#                            tmpFwdBackArgs = cp.deepcopy(args)
#                            tmpBkFwdArgs = cp.deepcopy(args)
#    
#                            tmpFwd = list(tmpFwdArgs)
#                            tmpBack = list(tmpBackArgs)
#                            tmpFwdBack = list(tmpFwdBackArgs)
#                            tmpBkFwd = list(tmpBkFwdArgs)
#                            
#                            
#                            if j._array.size == 1 and l._array.size == 1:
#                                dim1_index = 0
#                                dim2_index = 0
#                            elif l._array.size == 1:
#                                dim1_index = 0
#                                dim2_index = np.unravel_index(it.multi_index[0], j._array.shape)
#                            elif j._array.size == 1:
#                                dim1_index = np.unravel_index(it.multi_index[1], l._array.shape)
#                                dim2_index = 0
#                            else:
#                                dim1_index = np.unravel_index(it.multi_index[0], l._array.shape)
#                                dim2_index = np.unravel_index(it.multi_index[1], j._array.shape)
#                            
#                            tmpFwd[kKey][lKey]._array[dim1_index] += h
#                            tmpFwd[iKey][jKey]._array[dim2_index] += h
#                            
#                            tmpBack[kKey][lKey]._array[dim1_index] -= h
#                            tmpBack[iKey][jKey]._array[dim2_index] -= h
#                            
#                            tmpFwdBack[kKey][lKey]._array[dim1_index] += h
#                            tmpFwdBack[iKey][jKey]._array[dim2_index] -= h
#                            
#                            tmpBkFwd[kKey][lKey]._array[dim1_index] -= h
#                            tmpBkFwd[iKey][jKey]._array[dim2_index] += h
#                            
#                            intermediate_result[it.multi_index] = (fn(*tmpFwd) - fn(*tmpFwdBack) - fn(*tmpBkFwd) + fn(*tmpBack)) / (4*(h**2))
#                            
#                            #print(intermediate_result)
#                            #print(it.multi_index)
#                            
#                            it.iternext()
#                        #            print(intermediate_result)
#                        #                intermediate_result = 0.5*(intermediate_result + np.swapaxes(intermediate_result, 0, 1))
#                        inner_result.append(intermediate_result)
#                    #            print(inner_result)
#                result.append(np.hstack(inner_result))
#            
#        result = np.vstack(result)
#    
#    #    print(result)
#        return result

    def simple_fdiff_hessian(self, args, fn):
        h = 10e-4
        
        tmpFwdArgs = cp.deepcopy(args)
        tmpBackArgs = cp.deepcopy(args)
        tmpFwdBackArgs = cp.deepcopy(args)
        tmpBkFwdArgs = cp.deepcopy(args)
    
        tmpFwd = tmpFwdArgs
        tmpBack = tmpBackArgs
        tmpFwdBack = tmpFwdBackArgs
        tmpBkFwd = tmpBkFwdArgs
    
        intermediate_result = np.zeros([args.size, args.size])
        it = np.nditer(intermediate_result, flags=['multi_index'])
        while not it.finished:
            dim1_index = np.unravel_index(it.multi_index[0], args.shape)
            dim2_index = np.unravel_index(it.multi_index[1], args.shape)
            
            tmpFwd[dim1_index] += h
            tmpFwd[dim2_index] += h
            
            tmpBack[dim1_index] -= h
            tmpBack[dim2_index] -= h
            
            tmpFwdBack[dim1_index] += h
            tmpFwdBack[dim2_index] -= h
            
            tmpBkFwd[dim1_index] -= h
            tmpBkFwd[dim2_index] += h
            
            intermediate_result[it.multi_index] = (fn(tmpFwd) - fn(tmpFwdBack) - fn(tmpBkFwd) + fn(tmpBack)) / (4*(h**2))
            
            tmpFwd[dim1_index] -= h
            tmpFwd[dim2_index] -= h
            
            tmpBack[dim1_index] += h
            tmpBack[dim2_index] += h
            
            tmpFwdBack[dim1_index] -= h
            tmpFwdBack[dim2_index] += h
            
            tmpBkFwd[dim1_index] += h
            tmpBkFwd[dim2_index] -= h
            
            it.iternext()
        return intermediate_result
    
    def simple_fdiff_gradient(self, args, fn):
        h = 10e-4
    
        tmpFwdArgs = cp.deepcopy(args)
        tmpBackArgs = cp.deepcopy(args)
    
        tmpFwd = tmpFwdArgs
        tmpBack = tmpBackArgs
    
        intermediate_result = np.zeros(args.size)

        it = np.nditer(intermediate_result, flags=['multi_index'])
        while not it.finished:
            dim2_index = np.unravel_index(it.multi_index[0], args.shape)

            tmpFwd[dim2_index] += h
            tmpBack[dim2_index] -= h

            intermediate_result[it.multi_index] = (fn(tmpFwd) - fn(tmpBack)) / (2*h)

            tmpFwd[dim2_index] -= h
            tmpBack[dim2_index] += h

            it.iternext()
        return intermediate_result



#    def ugly_gradient(self, dma, dmb, ba, bb, mua, mub):
#        """This function calculates the gradient of the lagrangian with pauli exclusion and normalization constraints
#
#            The inputs are each an array of two objects, corresponding to alpha and beta spin.
#            - D is an array containing two instances of a oneBody density matrix
#            - B is an array containing two instances of a oneBody lagrangian multiplier matrix
#            - Mu is an array containing two scalars lagrangian multipliers
#
#        """
#        #allocate arrays for each block of matrix
#
#        D = [dma, dmb]
#        B = [ba,bb]
#        Mu = [mua,mub]
#
#        size_alpha = D[0]._array.size
#        size_beta = D[0]._array.size
#
#        grad = np.zeros(2*size_alpha + 2*size_beta + 2) #size of alpha and beta versions don't necessarily match.
#        grad_block=[]
#
#        for i in [0,1]:
#
#            dLdD = self.dVee.dD(D)._array
#
#            dm = D[i]._array
#            b = B[i]._array
#            mu = Mu[i]
#            t = self.T._array
#            v = self.V[i]._array
#            s = self.S._array
#            n = self.N[i]
#
#            #dL/dD block
#            dLdD += t - v - mu._array*np.eye(dm.shape[0])
#
#            if self._pauli_constraint(D[i], B[i]) < 0 or self.debug:
#                #print("Gradient tripped Pauli")
#                s2 = np.dot(s,s)
#                b2 = np.dot(b,b)
#                inside = s2 - np.dot(np.dot(s2,dm),s) - np.dot(np.dot(s,dm), s2)
#                outside = np.dot(b2,inside)
#                dLdD = dLdD - outside
#
#                #dL/dB block
#                inside = 0
#                inside = np.dot(np.dot(s,dm),s) - np.dot(np.dot(np.dot(np.dot(s,dm),s),dm),s)
#                dLdB = -2.*np.dot(inside,b)
#            else:
#                dLdB = np.zeros_like(b)
#            #dL/d_mu block
#            dLdMu = n - np.trace(np.dot(s,dm))
#
#            #assemble gradient in vector form
#            grad[size_alpha*i:(size_alpha+size_beta*i)] = dLdD.ravel()
#            b_offset = size_alpha+size_beta
#            grad[b_offset+size_alpha*i:(b_offset+size_alpha+size_beta*i)] = dLdB.ravel()
#            grad[(-2 + i)] = dLdMu
#
#            grad_block[len(grad_block):] = [dLdD, dLdB]
#
#        return grad


#    def gradient(self, D, B, Mu):
#        """This function calculates the gradient of the lagrangian with pauli exclusion and normalization constraints
#
#            The inputs are each an array of two objects, corresponding to alpha and beta spin.
#            - D is an array containing two instances of a oneBody density matrix
#            - B is an array containing two instances of a oneBody lagrangian multiplier matrix
#            - Mu is an array containing two scalars lagrangian multipliers
#
#        """
#        #allocate arrays for each block of matrix
#
#        size_alpha = D[0]._array.size
#        size_beta = D[0]._array.size
#
#        grad = np.zeros(2*size_alpha + 2*size_beta + 2) #size of alpha and beta versions don't necessarily match.
#        grad_block=[]
#
#        for i in [0,1]:
#
#            dLdD = self.dVee.dD(D)._array
#
#            dm = D[i]._array
#            b = B[i]._array
#            mu = Mu[i]
#            t = self.T._array
#            v = self.V[i]._array
#            s = self.S._array
#            n = self.N[i]
#
#            #dL/dD block
#            dLdD += t - v - mu._array*np.eye(dm.shape[0])
#
#            if self._pauli_constraint(D[i], B[i]) < 0 or self.debug:
#                #print("Gradient tripped Pauli")
#                s2 = np.dot(s,s)
#                b2 = np.dot(b,b)
#                inside = s2 - np.dot(np.dot(s2,dm),s) - np.dot(np.dot(s,dm), s2)
#                outside = np.dot(b2,inside)
#                dLdD = dLdD - outside
#
#                #dL/dB block
#                inside = 0
#                inside = np.dot(np.dot(s,dm),s) - np.dot(np.dot(np.dot(np.dot(s,dm),s),dm),s)
#                dLdB = -(np.dot(inside,b) + np.dot(b,inside))
##                print(dLdB)
#            else:
#                dLdB = np.zeros_like(b)
#            #dL/d_mu block
#            dLdMu = n - np.trace(np.dot(s,dm))
#
#            #assemble gradient in vector form
#            grad[size_alpha*i:(size_alpha+size_beta*i)] = dLdD.ravel()
#            b_offset = size_alpha+size_beta
#            grad[b_offset+size_alpha*i:(b_offset+size_alpha+size_beta*i)] = dLdB.ravel()
#            grad[(-2 + i)] = dLdMu
#
#            grad_block[len(grad_block):] = [dLdD, dLdB]
#
#        return grad

    def sym_gradient(self, D, B, Mu):
        """This function calculates the gradient of the lagrangian with pauli exclusion and normalization constraints

            The inputs are each an array of two objects, corresponding to alpha and beta spin.
            - D is an array containing two instances of a oneBody density matrix
            - B is an array containing two instances of a oneBody lagrangian multiplier matrix
            - Mu is an array containing two scalars lagrangian multipliers

        """
        #allocate arrays for each block of matrix

        size_alpha = D[0]._array.size #TODO: remove code for distinct alpha/beta sizes 
        size_beta = D[1]._array.size

        grad = np.zeros(2*size_alpha + 2*size_beta + 2) #size of alpha and beta versions don't necessarily match.


        for i in [0,1]:
            
            dLdD = self.dVee.dD(D)._array

            dm = D[i]._array
            b = B[i]._array
            mu = Mu[i]
            t = self.T._array
            v = self.V[i]._array
            s = self.S._array
            n = self.N[i]

            #dL/dD block
            dLdD += t - v - mu._array*s

            s2 = np.dot(s,s)
            b2 = np.dot(b,b)
            inside = s2 - np.dot(np.dot(s2,dm),s) - np.dot(np.dot(s,dm), s2)
            outside = np.dot(b2,inside)
            dLdD = dLdD - outside

            #dL/dB block
            inside = np.dot(np.dot(s,dm),s) - np.dot(np.dot(np.dot(np.dot(s,dm),s),dm),s)
            dLdB = -(np.dot(inside,b) + np.dot(b.T,inside))
#                print(dLdB)

            #dL/d_mu block
            dLdMu = n - np.trace(np.dot(s,dm))

#            #symmetrize
#            dLdD = 0.5*(dLdD + np.swapaxes(dLdD, 0, 1))
#            dLdB = 0.5*(dLdB + np.swapaxes(dLdB, 0, 1))

            #assemble gradient in vector form
            grad[size_alpha*i:(size_alpha+size_beta*i)] = dLdD.ravel()
            b_offset = size_alpha+size_beta
            grad[b_offset+size_alpha*i:(b_offset+size_alpha+size_beta*i)] = dLdB.ravel()
            grad[(-2 + i)] = dLdMu

        return grad

    def sym_hessian(self, D, B, Mu):
        #allocate matrix
        size_alpha = D[0]._array.shape[0]
        size_beta = D[1]._array.shape[0]
        hess = np.zeros([2*(size_alpha**2) + 2*(size_beta**2) + 2, 2*(size_alpha**2) + 2*(size_beta**2) + 2])

        for i in [0,1]:

            dm = cp.deepcopy(D[i]._array)
            b = cp.deepcopy(B[i]._array)
            s = cp.deepcopy(self.S._array)

            dLdDD = self.dVee.dD2(D[i])._array
            dLdBB = np.zeros_like(dLdDD)
            dLdDB = np.zeros_like(dLdDD)

#                print ("evaluating pauli block")
            dLdDD = self._calc_dLdDD(dLdDD, b)
            dLdDB = self._calc_dLdDB(dLdDB, dm, b)
            dLdBB = self._calc_dLdBB(dLdBB, dm)

            dLdDMu = -s.ravel()
#            dLdDMu = -np.ones(size_alpha**2)
            dLdDaDb = self.dVee.dDaDb(D[i])._array

            #symmetrize
            dLdDD = 0.25*(dLdDD + np.swapaxes(dLdDD, 0,1) + np.swapaxes(dLdDD,2,3) + np.swapaxes(np.swapaxes(dLdDD,0,1),2,3))
            dLdDB = 0.25*(dLdDB + np.swapaxes(dLdDB, 0,1) + np.swapaxes(dLdDB,2,3) + np.swapaxes(np.swapaxes(dLdDB,0,1),2,3))
            dLdBB = 0.25*(dLdBB + np.swapaxes(dLdBB, 0,1) + np.swapaxes(dLdBB,2,3) + np.swapaxes(np.swapaxes(dLdBB,0,1),2,3))
            dLdDaDb = 0.25*(dLdDaDb + np.swapaxes(dLdDaDb, 0,1) + np.swapaxes(dLdDaDb,2,3) + np.swapaxes(np.swapaxes(dLdDaDb,0,1),2,3))

            #reshape
            dLdDaDb = dLdDaDb.reshape([dLdDaDb.shape[0]**2, -1])
            dLdDD = dLdDD.reshape([dLdDD.shape[0]**2, -1])
            dLdDB = dLdDB.reshape([dLdDB.shape[0]**2, -1])
            dLdBB = dLdBB.reshape([dLdBB.shape[0]**2, -1])

            hess = self._assemble_hessian(hess, size_alpha, size_beta, dLdDD, dLdDaDb, dLdDB, dLdDMu, dLdBB, i)

        return hess

#    def hessian(self, D, B, Mu):
#        #allocate matrix
#        size_alpha = D[0]._array.shape[0]
#        size_beta = D[1]._array.shape[0]
#        hess = np.zeros([2*(size_alpha**2) + 2*(size_beta**2) + 2, 2*(size_alpha**2) + 2*(size_beta**2) + 2])
#
#        for i in [0,1]:
#
#            dm = cp.deepcopy(D[i]._array)
#            b = cp.deepcopy(B[i]._array)
#            s = cp.deepcopy(self.S._array)
#
#            dLdDD = self.dVee.dD2(D[i])._array
#            dLdBB = np.zeros_like(dLdDD)
#            dLdDB = np.zeros_like(dLdDD)
#            dLdBD = np.zeros_like(dLdDD)
#
#            if self._pauli_constraint(D[i], B[i]) < 0 or self.debug:
#                print ("evaluating pauli block")
#                dLdDD = self._calc_dLdDD(dLdDD, b)
#                dLdDB = self._calc_dLdDB(dLdDB, dm, b)
#                dLdBD = self._calc_dLdBD(dLdBD, dm, b)
#                dLdBB = self._calc_dLdBB(dLdBB, dm)
#
#            dLdDMu = -s.ravel()
#            dLdDaDb = self.dVee.dDaDb(D[i])._array
#            dLdDaDb = dLdDaDb.reshape([dLdDaDb.shape[0]**2, -1])
#            dLdDD = dLdDD.reshape([dLdDD.shape[0]**2, -1])
#            dLdDB = dLdDB.reshape([dLdDB.shape[0]**2, -1])
#            dLdBD = dLdBD.reshape([dLdBD.shape[0]**2, -1])
#            dLdBB = dLdBB.reshape([dLdBB.shape[0]**2, -1])
#
#            hess = self._assemble_hessian(hess, size_alpha, size_beta, dLdDD, dLdDaDb, dLdDB, dLdBD, dLdDMu, dLdBB, i)
#
#        return hess

    def _assemble_hessian(self, hess, size_alpha, size_beta, dLdDD, dLdDaDb, dLdDB, dLdBD, dLdDMu, dLdBB, i):
        #assemble hessian
        size_alpha2 = size_alpha**2
        size_beta2 = size_beta**2

        H_size = [0, size_alpha2, size_beta2, size_alpha2, size_beta2, 1,1]
        H_off = np.cumsum(H_size)

        #first/second column
        hess[H_off[0 +i]:H_off[1 +i], H_off[0 +i]:H_off[1 +i]] = dLdDD
        hess[H_off[0 +i]:H_off[1 +i], H_off[1 -i]:H_off[2 -i]] = dLdDaDb
        hess[H_off[2 +i]:H_off[3 +i], H_off[0 +i]:H_off[1 +i]] = dLdDB
        hess[H_off[4 +i]:H_off[5 +i], H_off[0 +i]:H_off[1 +i]] = dLdDMu

        #third/fourth column
        hess[H_off[0 +i]:H_off[1 +i], H_off[2 +i]:H_off[3 +i]] = dLdBD
        hess[H_off[2 +i]:H_off[3 +i], H_off[2 +i]:H_off[3 +i]] = dLdBB

        #fifth/sixth columns
        hess[H_off[0 +i]:H_off[1 +i], H_off[4 +i]:H_off[5 +i]] = np.reshape(dLdDMu,[dLdDMu.size,1])

        return hess


    def _calc_dLdDD(self, dLdDD, b): #TODO: refactor to use Toon's framework
        b2 = np.dot(b,b)
        s = self.S._array
        inside = np.dot(np.dot(b2, s), s)
        it = np.nditer(inside, flags=['multi_index'])
        while not it.finished:
            dLdDD[it.multi_index[0], :,:, it.multi_index[1]] += it*s #-[SB2S]_ad * [S]_bc
            dLdDD[:,it.multi_index[0],it.multi_index[1],:] += it*s
            it.iternext()
        return dLdDD

    def _calc_dLdDB(self, dLdDB, dm, b): #TODO: refactor to use Toon's framework
        s = self.S._array
        
        bs = np.dot(b,s)
        it = np.nditer(bs, flags=['multi_index'])
        while not it.finished:
            dLdDB[it.multi_index[0],:,:,it.multi_index[1]] -= it*s - it*np.dot(np.dot(s,dm),s)
            it.iternext()

        bsds = np.dot(np.dot(np.dot(b,s),dm),s)
        it = np.nditer(bsds, flags=['multi_index'])
        while not it.finished:
            dLdDB[it.multi_index[0],:,:,it.multi_index[1]] += it*s
            it.iternext()


#        it = np.nditer(s, flags=['multi_index'])
#        while not it.finished:
#            dLdDB[:,it.multi_index[0],it.multi_index[1],:] -= it*(np.dot(b,s)) - it*(np.dot(np.dot(np.dot(b,s),dm),s))
#            it.iternext()
#
#        sds = np.dot(np.dot(s,dm),s)
#        it = np.nditer(sds, flags=['multi_index'])
#        while not it.finished:
#            dLdDB[:,it.multi_index[0],it.multi_index[1],:] += it*(s)
#            it.iternext()
        
        sb = np.dot(s,b)
        it = np.nditer(sb, flags=['multi_index'])
        while not it.finished:
            dLdDB[:,it.multi_index[0],it.multi_index[1],:] -= it*s - it*(np.dot(np.dot(s,dm),s))
            it.iternext()
            
        sdsb = np.dot(np.dot(np.dot(s,dm),s),b)
        it = np.nditer(sdsb, flags=['multi_index'])
        while not it.finished:
            dLdDB[:,it.multi_index[0],it.multi_index[1],:] += it*s
            it.iternext()
        return dLdDB
    
    def _calc_dLdBD(self, dLdBD, dm, b):
        dLdBD = self._calc_dLdDB(dLdBD,dm, b).T
#        dLdBD = np.swapaxes(dLdBD, 1, 3)
#        dLdBD = np.swapaxes(dLdBD, 0, 2)
        
        return dLdBD

    def _calc_dLdBB(self, dLdBB, dm): #TODO: refactor to use Toon's framework
        s = self.S._array
        temp = np.zeros_like(dLdBB[:,:,0,0])
        for j in np.arange(dLdBB.shape[0]):
            temp = np.dot(np.dot(s,dm),s) - np.dot(np.dot(np.dot(np.dot(s,dm),s),dm),s)
#            dLdBB[j,:,:,j] -= temp
            dLdBB[:,j,:,j] -= temp
#            print(dLdBB)
        return dLdBB

    def _pauli_constraint(self, D, B):
        dm = D._array
        b = B._array
        s = self.S._array

        inside = np.dot(np.dot(s,dm),s) - np.dot(np.dot(np.dot(np.dot(s,dm),s),dm),s)
        result = np.trace(np.dot(np.dot(b,b.T),inside))
#        print("pauli condition,", result)
#        result = np.sum(np.dot(b,inside)*b)
#        if np.abs(np.sum(np.dot(b,inside)*b) - np.trace(np.dot(np.dot(b,inside),b))) > 10e-7:
#            print("DIFFERING trace/sum", np.sum(np.dot(b,inside)*b),np.trace(np.dot(np.dot(b,b),inside)))

#        inside = np.eye(dm.shape[0]) - np.dot(dm,s)
#        result = np.dot(np.dot(np.dot(np.dot(b, s), dm), s), b)*inside
#        result = np.dot(np.dot(np.dot(np.dot(b, b), s), dm), s)*inside
        if (np.abs(np.trace(np.dot(np.dot(b.T,inside),b)) - np.trace(np.dot(np.dot(b,b.T),inside))) > 10e-6):
            print("rounding failure")
            print(np.trace(np.dot(np.dot(b,inside),b)))
            print(np.trace(np.dot(np.dot(b,b),inside)))
            print("dm",dm)
            print("s",s)
            print("b",b)
            assert (np.abs(np.trace(np.dot(np.dot(b.T,inside),b)) - np.trace(np.dot(np.dot(b,b.T),inside))) < 1e-5).all()
        return result
    
    def _new_pauli_constraint(self, D, B):
        dm = D._array
        b = B._array
        s = self.S._array
#        p = P._array

#        inside = np.dot(np.dot(s,dm),s) - np.dot(np.dot(np.dot(np.dot(s,dm),s),dm),s) - np.dot(p,p.T)
        inside = np.dot(np.dot(s,dm),s) - np.dot(np.dot(np.dot(np.dot(s,dm),s),dm),s)
        result = np.trace(np.dot(b,inside))
#        print("pauli condition,", result)
#        result = np.sum(np.dot(b,inside)*b)
#        if np.abs(np.sum(np.dot(b,inside)*b) - np.trace(np.dot(np.dot(b,inside),b))) > 10e-7:
#            print("DIFFERING trace/sum", np.sum(np.dot(b,inside)*b),np.trace(np.dot(np.dot(b,b),inside)))

#        inside = np.eye(dm.shape[0]) - np.dot(dm,s)
#        result = np.dot(np.dot(np.dot(np.dot(b, s), dm), s), b)*inside
#        result = np.dot(np.dot(np.dot(np.dot(b, b), s), dm), s)*inside
        if (np.abs(np.trace(np.dot(inside,b)) - np.trace(np.dot(b,inside))) > 10e-6):
            print("rounding failure")
            print(np.trace(np.dot(np.dot(b,inside),b)))
            print(np.trace(np.dot(np.dot(b,b),inside)))
            print("dm",dm)
            print("s",s)
            print("b",b)
            assert (np.abs(np.trace(np.dot(np.dot(b.T,inside),b)) - np.trace(np.dot(np.dot(b,b.T),inside))) < 1e-5).all()
        return result
    
    def _1body_symmetrize(self,A):
        result = A + np.swapaxes(A, 0, 1) - A*np.eye(A.shape[0])
        return result
    
    def _2body_symmetrize(self, A): #TODO: incomplete code
        result = A + np.swapaxes(A, 0,1) + np.swapaxes(A,2,3) + np.swapaxes(np.swapaxes(A,0,1),2,3)
        it = np.nditer(result[0], flags=['multi_index'])
        while not it.finished:
            a = it.multi_index[0]
            c = it.multi_index[2]
            result[:,:,c,c] -= A[:,:,c,c] + np.swapaxes(A, 0,1)[:,:,c,c] 
            result[a,a,:,:] -= A[a,a,:,:] + np.swapaxes(A,2,3)[a,a,:,:] 
            
            it.iternext()
        
        result -= A[:,:,]
        return 
    
    def _idem_objective(self, b):
        result = np.trace(np.dot(np.dot(b,b.T),self.inside)) / np.trace(np.dot(b,b.T))
        return result
    
    def _rebuild_B(self,dx, b):
        dx = np.reshape(dx, b.shape)
        b += dx
        
        return b
    
    def _minimize_B(self, D, B, conditionNumber):
        dm = D._array
        b = B._array
        s = self.S._array
        
        bOld = cp.deepcopy(b)
        newtInst = newt()

        self.inside = np.dot(np.dot(s,dm),s) - np.dot(np.dot(np.dot(np.dot(s,dm),s),dm),s)
        bNew = newtInst.minimize(b, self.simple_fdiff_hessian, self.simple_fdiff_gradient, self._rebuild_B, lg = self, fn = self._idem_objective, conditionNumber = conditionNumber)
        #scale        
        b = b*np.sqrt(np.trace(np.dot(bOld, bOld))/np.trace(np.dot(bNew,bNew)))
        
        return b