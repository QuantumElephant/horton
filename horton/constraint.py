# -*- coding: utf-8 -*-
# Horton is a Density Functional Theory program.
# Copyright (C) 2011-2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--

from horton import *
import numpy as np

class Constraint(object):
    def __init__(self, sys, C, L, select):
        self.sys = sys
        self.L = L #a list or a single element
        self.select = select
        self.C = C
        print "New constraint initialized with C:" + str(C)

    def lagrange(self,D, Mul):
        raise NotImplementedError
    def self_gradient(self,D):
        raise NotImplementedError
    def D_gradient(self, D, Mul):
        raise NotImplementedError
    
    @staticmethod
    def integrate(system, grid, potential, suffix="L"):
        term = CustomGridFixedTerm(grid, potential.squeeze(), suffix)
        operator = term.get_operator(system)[0]._array
        return operator
    
    def next(self):
        if self.steps_array is None or self.steps_array.size == 0:
            return False
        print "Setting old C: " +str(self.C)
        self.C = self.steps_array[0]
        print "to new C: " + str(self.C)
        self.steps_array = self.steps_array[1:]
        return True
        
class LinearConstraint(Constraint):
    def __init__(self, sys, C, L, select, C_init = None, D = None, steps = 10):
        if D is not None and C_init is None:
            C_init = np.trace(reduce(np.dot,[L,sys.get_overlap()._array,D]))
        if C_init is not None:
            C_final = C
            self.steps_array = np.linspace(C_init, C_final, steps)[1:]
        else:
            C_init = C
        super(LinearConstraint, self).__init__(sys, C_init, L, select)
    
    def lagrange(self, D, mul):
        S = self.sys.get_overlap()._array #don't have lg.toNumpy() yet.
        P = np.dot(S,D)
        mul = mul.squeeze() #TODO: remove me
        return -mul*(np.dot(self.L.ravel(), P.ravel()) - self.C)
    def self_gradient(self, D):
        S = self.sys.get_overlap()._array #don't have lg.toNumpy() yet.
        P = np.dot(S,D)
        return self.C - np.dot(self.L.ravel(), P.ravel()) #Breaks compatibility with pre-constraint rewrite code. Original form below.
#        return self.C - np.trace(np.dot(P, self.L))
    def D_gradient(self, D, Mul):
        S = self.sys.get_overlap()._array #don't have lg.toNumpy() yet.
        Mul = Mul.squeeze()
        SL = np.dot(S, self.L)
        
        return -Mul*0.5*(SL + SL.T) #Should this always be negative?
    
class QuadraticConstraint(Constraint):
    def __init__(self, sys, C, L, select, C_init = None, D = None, steps = 10):
        assert len(L) == 2
        self.sys = sys
        if D is not None and C_init is None:
            S = self.sys.get_overlap()._array
            P = np.dot(S, D)
            C_init = np.trace(reduce(np.dot,[L[0], S, D, L[1], S, D]))
        if C_init is not None:
            C_final = C/2.
            self.steps_array = np.linspace(C_init/2., C_final, steps)[1:]
        else:
            C_init = C
        
        super(QuadraticConstraint, self).__init__(sys, C_init/2., L, select)
    def lagrange(self, D, mul):
        S = self.sys.get_overlap()._array #don't have lg.toNumpy() yet.
        P = np.dot(S,D)
        LaSD = np.dot(self.L[0], P)
        LbSD = np.dot(self.L[1], P)
        mul = mul.squeeze() #TODO: remove me
        
        assert np.abs(np.dot(LaSD.ravel(), LbSD.T.ravel()) - np.dot(LbSD.ravel(), LaSD.T.ravel())) < 1e-8, (np.dot(LaSD.ravel(), LbSD.T.ravel()),np.dot(LbSD.ravel(), LaSD.T.ravel()))
        
        return -mul*(np.dot(LaSD.ravel(), LbSD.T.ravel()) - self.C) #need the transpose to sum properly!
#         return -Mul*(np.trace(np.dot(LaSD,LbSD)) - self.C)
    def self_gradient(self, D):
        S = self.sys.get_overlap()._array #don't have lg.toNumpy() yet.
        P = np.dot(S,D)
        LaSD = np.dot(self.L[0], P)
        LbSD = np.dot(self.L[1], P)
        
        return self.C - np.dot(LaSD.ravel(), LbSD.T.ravel()) #need the transpose to sum properly!
#        return self.C*2 - np.trace(np.dot(P, self.L)) 
    def D_gradient(self, D, Mul):
        S = self.sys.get_overlap()._array #don't have lg.toNumpy() yet.
        Mul = Mul.squeeze()
        P = np.dot(S, D)
        L0SDL1S = np.dot(np.dot(np.dot(self.L[0],P),self.L[1]),S)
        L1SDL0S = np.dot(np.dot(np.dot(self.L[1],P),self.L[0]),S)
        
        result = -Mul*0.5*(L1SDL0S + L1SDL0S.T + L0SDL1S + L0SDL1S.T)
#        result = -Mul*(L1SDL0S + L0SDL1S).T
#        assert (np.abs(result - result.T) < 1e-8).all()
        
        return result #Should this always be negative?
    
#     def combGrad(self, x): #testing
#         Mul = x[-1]
#         D = x[:-1].reshape(7,7)
#         
#         dLdD = self.D_gradient(D, Mul)
# #        assert (np.abs(dLdD - dLdD.T) < 1e-8).all()
#         
#         result = [dLdD.ravel(), self.self_gradient(D)]
#         result = np.hstack(result)
#         
#         return result
#     
#     def reshapeX(self,x):
#         Mul = x[-1]
#         D = x[:-1].reshape(7,7)
#         
#         result = self.lagrange(D, Mul)
# #        print result
#         
#         return result
