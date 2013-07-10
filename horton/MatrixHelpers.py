import numpy as np
import pylab
# from horton.matrix import DenseOneBody
from horton.matrix import *

class MatrixHelpers(object):
    def __init__(self, sys, shapes):
        self.sys = sys
        self.shapes = shapes
        self.offsets = self.calc_offsets()
        
    def matToVec(self, *args):
        raise NotImplementedError
    def vecToMat(self, *args):
        raise NotImplementedError
    def calc_offsets(self):
        raise NotImplementedError
    def new_one_body(self):
        return self.sys.lf.create_one_body(self.shapes[0])
    def new_one_body_from(self, A):
        return self.sys.lf.create_one_body_from(A)
    def toOneBody(self, *args):
        result = []
        for i in args:
            assert isinstance(i,np.ndarray)
            if i.size == 1:
                result.append(i)
            else:
                temp = self.sys.lf.create_one_body(i.shape[0])
                temp._array = i
                result.append(temp)
            
        if len(result) == 1:
            return result[0]
        return result
    
    def toNumpy(self, *args):
        result = []
        for i in args:
            if isinstance(i, DenseOneBody):
                result.append(i._array)
            else:
                result.append(i)
         
        if len(result) == 1:
            return result[0]   
        return result
    
    def symmetrize(self, *args):
        result = []
        for i in args:
            if isinstance(i, np.ndarray):
                result.append(0.5*(i+i.T))
            else:
                i.isymmetrize()
                result.append(i)
        return result
    
    def check_sym(self, *args):
        def plot_mat(self, mat):
            pylab.matshow(mat)
            pylab.show()
        
        for i in args:
            if i.size == 1:
                continue
            
            shortDim = np.min(i.shape)
            if shortDim != np.max(i.shape):
                print "truncating matrix for plotting"
            symerror = np.abs(i[:,:shortDim] - i.T[:shortDim,:])
            if not (symerror < 1e-8).all():
                print "sym:", args
            assert (symerror < 1e-8).all(), (np.vstack(np.where(symerror > 1e-8)).T, symerror,np.sort(symerror, None)[-20:], self.plot_mat(symerror > 1e-8))

class FullMatrixHelpersOLD(MatrixHelpers):
    def calc_offsets(self):
        result = [0] + [n**2 for n in self.shapes]
        result = np.cumsum(result)
        return result
    
    def matToVec(self, *args):
        result = [i.ravel() for i in args]    
        x = np.hstack(result)
        return x
    
    def vecToMat(self,x):
        args = []
        for i in np.arange(len(self.offsets)-1):
            args.append(x[self.offsets[i]:self.offsets[i+1]].reshape([self.shapes[i], self.shapes[i]]))
        return args

class FullMatrixHelpers(MatrixHelpers):
    def calc_offsets(self):
        result = [0] + [n**2 for n in self.shapes]
        result = np.cumsum(result)
        return result
    
    def matToVec(self, *args):
        result = [i.ravel() for i in args]    
        x = np.hstack(result)
        return x
    
    def vecToMat(self,x):
        args = []
        for i in np.arange(len(self.offsets)-1):
            temp = x[self.offsets[i]:self.offsets[i+1]].reshape([self.shapes[i], self.shapes[i]])
            if temp.size > 1: #not a singleton
                temp = self.new_one_body_from(temp)
            args.append(temp)
        return args

class TriuMatrixHelpersOLD(MatrixHelpers):
    def calc_offsets(self):
        result = [0] + [int((n + 1)*n/2.) for n in self.shapes]
        result = np.cumsum(result)
        return result
        
    def matToVec(self, *args):
        """ Takes an array of dense matrices and returns a vector of the upper triangular portions.
            Does not check for symmetry first.
        """
        result = []
        for i in args:
            if i.size == 1:
                result.append(i.squeeze())
            else:
                ind = np.triu_indices_from(i)
                result.append(i[ind])
        x = np.hstack(result)
        return x
    
    def vecToMat(self,x):
        args = []
        for i in np.arange(len(self.offsets)-1):
            
            if self.shapes[i] == 1: #try to remove me
                args.append(x[self.offsets[i]:self.offsets[i+1]])
                continue
            
            ut = np.zeros([self.shapes[i], self.shapes[i]])
            ind = np.triu_indices_from(ut)
            ut[ind] = x[self.offsets[i]:self.offsets[i+1]]
            temp = 0.5*(ut + ut.T)
            args.append(temp)
        self.check_sym(*args)
        return args

class TriuMatrixHelpers(MatrixHelpers):
    def calc_offsets(self):
        result = [0] + [int((n + 1)*n/2.) for n in self.shapes]
        result = np.cumsum(result)
        return result
        
    def matToVec(self, *args):
        """ Takes an array of dense matrices and returns a vector of the upper triangular portions.
            Does not check for symmetry first.
        """
        result = []
        for i in args:
            if not isinstance(i, OneBody):
                result.append(i.squeeze())
            else:
                result.append(i.ravel())
        x = np.hstack(result)
        return x
    
    def vecToMat(self,x):
        args = []
        for i in np.arange(len(self.offsets)-1):
            if self.shapes[i] == 1: #try to remove me
                args.append(x[self.offsets[i]:self.offsets[i+1]])
            else:
                temp = self.new_one_body()
                vec = 0.5*x[self.offsets[i]:self.offsets[i+1]]
                temp.set_elements_from_vec(vec)
                temp.iscale_diag(2)
                args.append(temp)
        return args
