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

import numpy as np

class Constraint(object):
    def __init__(self, sys, C, L):
        self.sys = sys
        self.S = sys.get_overlap()._array #don't have lg.toNumpy() yet.
#        self.S = np.eye(7)
        self.C = C
        self.L = L #a list or a single element
        
    def lagrange(self,D, Mul):
        raise NotImplementedError
    def self_gradient(self,D):
        raise NotImplementedError
    def D_gradient(self, D, Mul):
        raise NotImplementedError
        
class LinearConstraint(Constraint):
    def lagrange(self, D, Mul):
        P = np.dot(self.S,D)
        Mul = Mul.squeeze() #TODO: remove me
        return -Mul*(np.dot(self.L.ravel(), P.ravel()) - self.C)
    def self_gradient(self, D):
        P = np.dot(self.S,D)
        return self.C - np.dot(P.ravel(), self.L.ravel()) #Breaks compatibility with pre-constraint rewrite code. Original form below.
#        return self.C - np.trace(np.dot(P, self.L))
    def D_gradient(self, D, Mul):
        Mul = Mul.squeeze()
        SL = np.dot(self.S, self.L)
        
        return -Mul*0.5*(SL + SL.T) #Should this always be negative?
    
class QuadraticConstraint(Constraint):
    def __init__(self, sys, C, L):
        assert len(L) == 2
        super(QuadraticConstraint, self).__init__(sys, C, L)
    def lagrange(self, D, Mul):
        P = np.dot(self.S,D)
        LaSD = np.dot(self.L[0], P)
        LbSD = np.dot(self.L[1], P)
        Mul = Mul.squeeze() #TODO: remove me
        
        assert np.abs(np.dot(LaSD.ravel(), LbSD.T.ravel()) - np.dot(LbSD.ravel(), LaSD.T.ravel())) < 1e-8, (np.dot(LaSD.ravel(), LbSD.T.ravel()),np.dot(LbSD.ravel(), LaSD.T.ravel()))
        
#        return -Mul*(np.dot(LaSD.ravel(), LbSD.T.ravel()) - self.C) #need the transpose to sum properly!
        return -Mul*(np.trace(np.dot(LaSD,LbSD)) - self.C)
    def self_gradient(self, D):
        P = np.dot(self.S,D)
        LaSD = np.dot(self.L[0], P)
        LbSD = np.dot(self.L[1], P)
        
        return self.C - np.dot(LaSD.ravel(), LbSD.T.ravel()) #need the transpose to sum properly!
#        return self.C - np.trace(np.dot(P, self.L)) 
    def D_gradient(self, D, Mul):
        Mul = Mul.squeeze()
        P = np.dot(self.S, D)
        L0SDL1S = np.dot(np.dot(np.dot(self.L[0],P),self.L[1]),self.S)
        L1SDL0S = np.dot(np.dot(np.dot(self.L[1],P),self.L[0]),self.S)
        
        result = -Mul*0.5*(L1SDL0S + L1SDL0S.T + L0SDL1S + L0SDL1S.T)
#        result = -Mul*(L1SDL0S + L0SDL1S).T
#        assert (np.abs(result - result.T) < 1e-8).all()
        
        return result #Should this always be negative?
    
    def combGrad(self, x): #testing
        Mul = x[-1]
        D = x[:-1].reshape(7,7)
        
        dLdD = self.D_gradient(D, Mul)
#        assert (np.abs(dLdD - dLdD.T) < 1e-8).all()
        
        result = [dLdD.ravel(), self.self_gradient(D)]
        result = np.hstack(result)
        
        return result
    
    def reshapeX(self,x):
        Mul = x[-1]
        D = x[:-1].reshape(7,7)
        
        result = self.lagrange(D, Mul)
#        print result
        
        return result