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

class LinearConstraint(Constraint):
    def lagrange(self, D, mul):
        S = self.sys.get_overlap()
        mul = mul.squeeze()
        lsd = self.sys.lf.create_one_body_eye()
        lsd.imuls(self.L, S, D)
        return -mul*(lsd.trace() - self.C)
    def self_gradient(self, D):
        S = self.sys.get_overlap()
        lsd = self.sys.lf.create_one_body_eye()
        lsd.imuls(self.L, S, D)
        return self.C - lsd.trace()
    def D_gradient(self, D, Mul):
        S = self.sys.get_overlap()
        Mul = Mul.squeeze()
        SL = self.sys.lf.create_one_body_eye()
        SL.imuls(S, self.L)
        SL.isymmetrize()
        SL.iscale(-Mul)
        return SL  #Should this always be negative?
    
class QuadraticConstraintOLD(Constraint):
    def __init__(self, sys, C, L, select):
        assert len(L) == 2
        
        super(QuadraticConstraint, self).__init__(sys, C, L, select)
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
