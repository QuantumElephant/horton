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
import copy as cp
import scipy.optimize as op
import scipy.linalg as lnag
from horton.matrix import DenseOneBody

from matplotlib import pyplot
from numpy import polynomial as poly

class newton(object):
    """This is the abstract base class for newton methods
    """
    def __init__(self):
        return

    def minimize(self, x0, H, g, rebuild, stop):
        raise NotImplementedError

class reallyStupidNewton(newton):
    """ The dumbest implementation of Newton. No trust radius etc.
        Solves Hx = -g using numpy's LU solver implementation, linalg.solve
    """

    def minimize(self,x0,H,g,rebuild=None,stop=10e-5, lg = None, fn = None, conditionNumber = 1e-8):
        """Takes an:
            - initial guess x0 of type np.array
            - function returning the hessian matrix at x of type np.array, H(x)
            - function returning the gradient vector at x of type np.array, g(x)

        Returns the:
            - np.array containing the x at which max(H*x) < stop
        """

#        print("before", lg._idem_objective(x0))

        x=x0
        for i in range(500):
#            print("iter", i)
            
            args = (x)
            if fn is not None:
                args = (x, fn)
            
            eigVal, eigVec = np.linalg.eigh(H(*args))
            H_inv = 0
            for i,eigVal_i in enumerate(eigVal):
                if np.abs(eigVal[i]) < conditionNumber:
                    if eigVal[i] == 0.0:
                        eigVal[i] = 1e8
                    else:
                        eigVal[i] = np.sign(eigVal[i])*1e8;# print("forcing large singular values")
                H_inv += (1./eigVal[i])*np.ma.outer(eigVec[:,i], eigVec[:,i])
            
            dx=np.linalg.solve(H_inv, -g(*args))
            if rebuild is not None:
                x = rebuild(dx,x)
            else:
                x = dx+x
            
            print(lg._idem_objective(x))    
            if lg._idem_objective(x) < 0:
                break
        return x
    
#    def conditioned_minimize(self, x0, H, g, rebuild=None, stop=10e-5, mask=None):
##        cutoff = 100
#        x=cp.deepcopy(x0)
#        if rebuild is not None:
#            lin_x = (x[0][0]._array.ravel(), x[0][1]._array.ravel(), x[1][0]._array.ravel(), 
#                 x[1][1]._array.ravel(), x[2][0]._array.ravel(), x[2][1]._array.ravel())
#            lin_x = cp.deepcopy(np.hstack(lin_x))
#        else: 
#            lin_x = x[0]
#        for i in range(1000):
#            print('iteration: ',i)
#            
#            H_inv = np.linalg.pinv(H(*x))
#            
#            dx=np.dot(H_inv, -g(*x))
#
#            if rebuild is not None:
#                x,lin_x = rebuild(dx,x,lin_x, mask)
#                print("d-d0", np.sqrt(np.sum((x[0][0]._array-x0[0][0]._array)*(x[0][0]._array-x0[0][0]._array))))
#            else:
#                x = dx+x
#            
#            if abs(dx).max() < stop:
#                break
#        return x, lin_x
    
#    def trustR_conditioned_minimize(self, x0, H, g, rebuild=None, stop=10e-5, mask=None, offset = None, lg = None, initialR = 1):
#        
#        xk=cp.deepcopy(x0)
#        if isinstance(xk[0][0], DenseOneBody):
#            lin_xk = (xk[0][0]._array.ravel(), xk[0][1]._array.ravel(), xk[1][0]._array.ravel(), 
#                 xk[1][1]._array.ravel(), xk[2][0]._array.ravel(), xk[2][1]._array.ravel())
#            lin_xk = cp.deepcopy(np.hstack(lin_xk))
#        else:
#            lin_xk = xk[0]
#            
#        trustR = initialR
#        
#        gk = g(*xk)
#        Hk = H(*xk)
#        
#        xk1 = cp.deepcopy(xk)
#        lin_xk1 = cp.deepcopy(lin_xk)
#        
#        for i in range(1000):
#            if i > 20:
#                lg.fine_tune = True
#            
#            print("\n")
##            print('iteration: ',i+1, "xk: ", xk1)
#            print('iteration: ',i+1)
#            eigVal, eigVec = np.linalg.eigh(Hk)
##            H_inv = np.linalg.pinv(Hk)
#            Hk_noshift = Hk; print("Hessian DB", Hk[12:16, 12:16]); print("pauli constraint", lg._pauli_constraint(xk[0][0], xk[1][0]))
#            Hk = 0
#            H_inv = 0
#            for i,eigVal_i in enumerate(eigVal):
##                eigVal[i] += np.sign(eigVal[i])*10e-4; print("offsetting all eigenvalues")
#                Hk += eigVal[i]*np.ma.outer(eigVec[:,i], eigVec[:,i]) 
#                if np.abs(eigVal[i]) < 1e-8:
##                    eigVal[i] = np.sign(eigVal[i])*10e-8; print("resetting eigenvalue")
#                    if eigVal[i] == 0.0:
#                        eigVal[i] = 1e8
#                    else:
#                        eigVal[i] = np.sign(eigVal[i])*1e8;# print("forcing large singular values")
#                
#
##                if offset is not None:
##                    Hk[offset[0]:offset[1], offset[0]:offset[1]] = self._shift_blockeig(Hk[offset[0]:offset[1], offset[0]:offset[1]], 10e-3)
##                    Hk[offset[1]:offset[2], offset[1]:offset[2]] = self._shift_blockeig(Hk[offset[1]:offset[2], offset[1]:offset[2]], 10e-3)
##                    Hk[offset[2]:offset[3], offset[2]:offset[3]] = self._shift_blockeig(Hk[offset[2]:offset[3], offset[2]:offset[3]], -10e-3)
##                    Hk[offset[3]:offset[4], offset[3]:offset[4]] = self._shift_blockeig(Hk[offset[3]:offset[4], offset[3]:offset[4]], -10e-3)
##                    self._check_hess_sign(Hk, offset)
#                
#                H_inv += (1./eigVal[i])*np.ma.outer(eigVec[:,i], eigVec[:,i])
#
#            dx=np.dot(H_inv, -gk)
#            dx_prev = np.zeros_like(dx)*10e10
#
#            xk1_trial = cp.deepcopy(xk)
#            lin_xk1_trial = cp.deepcopy(lin_xk)
#
#            if rebuild is not None:
#                if np.linalg.norm(dx) > trustR:
#                    print("Step too long. Using Sbar")
#                    sbar = self._solve_sbar(trustR, gk, Hk, eigVec, eigVal)
#                    dx = sbar
#                    dx_temp = dx
#                
#                xk1_trial,lin_xk1_trial= rebuild(dx,xk1_trial, lin_xk1_trial, mask)
#
#                gk1 = g(*xk1_trial)
#                
#                print("check sbar 3rd order", self._check_3rd_order(gk, gk1, Hk_noshift, dx))
##                print("check unit 3rd order", self._unit_step(xk, dx, H, g))
#                
#                if np.max(dx_temp) < np.min(1e-4, 1e-1*np.max(dx_prev)):
#                    print("accepting step", np.max(dx_temp))
#                    dx, trustR = self._trust_radius(trustR, gk, gk1, Hk_noshift, dx, True)
#                else:
#                    dx, trustR = self._trust_radius(trustR, gk, gk1, Hk_noshift, dx, False)
#                xk1,lin_xk1 = rebuild(dx,xk1,lin_xk1, mask)
#
##SELF-CONSISTENCY CHECKS
#                print("check occupation", self._check_occ(xk1, lg.S))
#                print("idempotency multiplier", self._check_idem(xk1, lg.S))
#                print("normalization multiplier", xk1[2][0]._array, xk1[2][1]._array)
##                self._check_symmetry(xk1)
#                
##                if isinstance(x[0][0], DenseOneBody):    
##                    print("d-d0", np.sqrt(np.sum((x[0][0]._array-x0[0][0]._array)*(x[0][0]._array-x0[0][0]._array))))
#                    
#            dx_prev = dx
#            gk = g(*xk1)
#            Hk = H(*xk1)
#            
#            if lg is not None:
#                print(lg.energy(xk1[0]))
#                
#            if abs(gk).max() < stop:
#                np.set_printoptions(precision = 4)
#                print(Hk)
#                break
#            
#            xk = xk1
#            lin_xk = lin_xk1
#        return xk1, lin_xk1
#    
#    def trustR_SR1_minimize(self, x0, g, H0, H0_inv, rebuild=None, stop=10e-5, mask=None, offset = None, lg = None, initialR = 1):
#        
#        xk=cp.deepcopy(x0)
#        if isinstance(xk[0][0], DenseOneBody):
#            lin_xk = (xk[0][0]._array.ravel(), xk[0][1]._array.ravel(), xk[1][0]._array.ravel(), 
#                 xk[1][1]._array.ravel(), xk[2][0]._array.ravel(), xk[2][1]._array.ravel())
#            lin_xk = cp.deepcopy(np.hstack(lin_xk))
#        else:
#            lin_xk = xk[0]
#            
#        trustR = initialR
#        
#        gk = g(*xk)
#        H_inv = H0_inv
#        Hk = H0
#        
#        xk1 = cp.deepcopy(xk)
#        lin_xk1 = cp.deepcopy(lin_xk)
#        
#        for i in range(10000):
#            
#            print("\n")
##            print('iteration: ',i+1, "xk: ", xk1)
#            print('iteration: ',i+1)
#
#            dx=np.dot(H_inv, -gk)
#            dx_prev = np.zeros_like(dx)*10e10
#
#            xk1_trial = cp.deepcopy(xk)
#            lin_xk1_trial = cp.deepcopy(lin_xk)
#
#            if rebuild is not None:
#                eigVal, eigVec = np.linalg.eigh(Hk)
#                
#                if np.linalg.norm(dx) > trustR:
#                    print("Step too long. Using Sbar")
#                    sbar = self._solve_sbar(trustR, gk, Hk, eigVec, eigVal)
#                    dx = sbar
#                    dx_temp = dx
#                
#                xk1_trial,lin_xk1_trial= rebuild(dx,xk1_trial, lin_xk1_trial, mask)
#
#                gk1 = g(*xk1_trial)
#                
#                print("check sbar 3rd order", self._check_3rd_order(gk, gk1, Hk, dx))
##                print("check unit 3rd order", self._unit_step(xk, dx, H, g))
#                
#                if np.max(dx_temp) < np.min(1e-4, 1e-1*np.max(dx_prev)):
#                    print("accepting step", np.max(dx_temp))
#                    boolAcceptStep = True
#                    dx, trustR = self._trust_radius(trustR, gk, gk1, Hk, dx, boolAcceptStep)
#                else:
#                    boolAcceptStep = False
#                    dx, trustR = self._trust_radius(trustR, gk, gk1, Hk, dx, boolAcceptStep)
#                xk1,lin_xk1 = rebuild(dx,xk1,lin_xk1, mask)
#
##SELF-CONSISTENCY CHECKS
#                print("check occupation", self._check_occ(xk1, lg.S))
#                print("idempotency multiplier", self._check_idem(xk1, lg.S))
#                print("normalization multiplier", xk1[2][0]._array, xk1[2][1]._array)
##                self._check_symmetry(xk1)
#                
#            dx_prev = dx
#            gk1 = g(*xk1)
#            if np.max(np.abs(dx)) > 0: #only update H if trust radius isn't zeroed out  
#                Hk1 = Hk + self.SR1_Hk1_dx(Hk, gk1, gk, dx)
#                H_inv1 = H_inv + self.SR1_Hk1_inv_dx(H_inv, gk1, gk, dx)
#                Hk = Hk1
#                H_inv = H_inv1
##                Hk = np.linalg.pinv(H_inv)
##                H_inv = np.linalg.pinv(Hk)
#            
#            if lg is not None:
#                print(lg.energy(xk1[0]))
#                
#            if abs(gk).max() < stop:
#                np.set_printoptions(precision = 4)
#                print(Hk)
#                break
#           
#            gk = gk1
#            xk = xk1
#            lin_xk = lin_xk1
#        return xk1, lin_xk1
    
    def line_SR1_minimize(self, x0, g, H0_inv, rebuild, stop=10e-5, mask=None, offset = None, lg = None):
        
        xk=cp.deepcopy(x0)
        if isinstance(xk[0][0], DenseOneBody):
            lin_xk = (xk[0][0]._array.ravel(), xk[0][1]._array.ravel(), xk[1][0]._array.ravel(), 
                 xk[1][1]._array.ravel(), xk[2][0]._array.ravel(), xk[2][1]._array.ravel())
            lin_xk = cp.deepcopy(np.hstack(lin_xk))
        else:
            lin_xk = xk[0]
            
        gk = g(*xk)
        H_inv = H0_inv
        
        for i in range(10000):
            
            print("\n")
#            print('iteration: ',i+1, "xk: ", xk1)
            print('iteration: ',i+1)

            dx_full=np.dot(H_inv, -gk)
    
            xk1_full = cp.deepcopy(xk)
            xk1_full = rebuild(dx_full,xk1_full, mask=mask)
            gk1_full = g(*xk1_full)
            
            print("check full step finite diff 3rd order", self._check_3rd_order(gk, gk1_full, H_inv, dx_full))
            
            debug_step = False
            print("debug: choosing step length"); debug_step = True
            if np.linalg.norm(gk1_full) > 0.9*np.linalg.norm(gk) or debug_step: 
                dx_step, xk1_step = self._debug_line_search(dx_full, xk, g, rebuild, lg=lg)
                gk1_step = g(*xk1_step)
            else:
                print("acceptable decrease")
                xk1_step = xk1_full
                dx_step = dx_full
                gk1_step = gk1_full
        
#SELF-CONSISTENCY CHECKS
#            print("check sbar 3rd order", self._check_3rd_order(gk, gk1_step, H_inv, dx_step))
#            print("check occupation", self._check_occ(xk1_step, lg.S))
#            print("idempotency multiplier", self._check_idem(xk1_step, lg.S))
#            print("normalization multiplier", xk1_step[2][0]._array, xk1_step[2][1]._array)
#                self._check_symmetry(xk1)
            
#            gk1 = g(*xk1)
            
            
            if lg is not None:
                print("Energy", lg.energy(xk1_step[0]))
           
            print("check modified step finite diff 3rd order", self._check_3rd_order(gk, gk1_step, H_inv, dx_step))

            

            H_inv = H_inv + self.SR1_Hk1_inv_dx(H_inv, gk1_step, gk, dx_step)
            gk = gk1_step
            xk = xk1_step
            
            if np.linalg.norm(gk1_step) < stop:
                np.set_printoptions(precision = 4)
                print(H_inv)
                break
            
        return xk
    
    def _debug_line_search(self, dx_full, xk, g, rebuild, mask = None, lg=None):
        alpha = 0.05
        alpha_max = 4
        gradients = []
        
        step = np.arange(-alpha_max, alpha_max, alpha)
        
        for i in step:
            xk1_temp = cp.deepcopy(xk)
            xk1_temp = rebuild(dx_full*i, xk1_temp, mask = mask)
#            
            gradients.append(np.linalg.norm(g(*xk1_temp))**2)
            
        #polynomial interpretation
        coeffs = poly.polynomial.polyfit(step, gradients, 2)
#        eval_grad = poly.polyval(step, coeffs)
        
        min_poly = -coeffs[1]/(2*coeffs[2]) 
        
#        if np.abs(min_poly) > np.sqrt(xk[0][0]._array.shape[0])*0.1:
#            min_poly = np.sign(min_poly)*np.sqrt(xk[0][0]._array.shape[0])*0.1
        
        xk1_min = cp.deepcopy(xk)
        xk1_min = rebuild(dx_full*min_poly, xk1_min, mask = mask)
        grad_min = np.linalg.norm(g(*xk1_min))**2
        
        pyplot.plot(min_poly, grad_min, 'o')
#        pyplot.plot(step, eval_grad)
        pyplot.plot(step, gradients)
#        pyplot.show()
        
        print("min step norm", np.linalg.norm(dx_full*min_poly))
        print("minimizing alpha", min_poly)
        xk_result = cp.deepcopy(xk)
        dx_step = dx_full*min_poly
        
        return dx_step, rebuild(dx_step, xk_result, mask = mask)
    
    def _curvature_condition_check(self, gk, gk1, xk, dx, c2=0.9):
        cond2 = np.abs(np.dot(gk1, dx)) <= c2*np.abs(np.dot(gk,dx))
        
        return  cond2
    
    def SR1_Hk1_inv_dx(self, Hk_inv, gk1, gk, dx):
        r = 1e-5
        yk = gk1 - gk
        inside = dx - np.dot(Hk_inv,yk)
        
        if np.abs(np.dot(dx.T, inside)) < r*np.linalg.norm(dx)*np.linalg.norm(inside):
            print("skipped a SR1 inv update.")
            return np.zeros_like(Hk_inv)
        
        result = np.outer(inside, inside)/np.dot(inside.T,dx)
        return result
    
    def SR1_Hk1_dx(self, Hk, gk1, gk, dx):
        r = 1e-5
        yk = gk1 - gk
        inside = yk - np.dot(Hk,dx)
        
        if np.abs(np.dot(dx.T, inside)) < r*np.linalg.norm(dx)*np.linalg.norm(inside):
            print("skipped a SR1 update.")
            return np.zeros_like(Hk)
        
        result = np.outer(inside, inside)/np.dot(inside.T,dx)
        return result
    
#    def _cos_gradient(self, gk, gk1, gk1Pred):
#        diffG = gk1-gk
#        diffGPred = gk1Pred - gk
#        dotDiffG = np.dot(diffG, diffGPred)
#        
#        result = dotDiffG / (np.linalg.norm(diffG)*np.linalg.norm(diffGPred))
#        return result
#
#    def _mag_gradient(self, normGk, normGk1, normGk1Pred):
#        result = (normGk1 - normGk) / (normGk1Pred - normGk)
#        return result
#    
#    def _p10(self,d):
#        return 1.6424/d + 1.11/(d**2)
#    
#    def _p40(self,d):
#        return 0.064185/d + 0.0946/(d**2)
#    
#    def _sbar(self, eigVec, eigVal, gk, lambdaBar):
#        sbar = 0
#        for i in range(eigVec.shape[1]):
##            if np.abs(eigVal[i]) < 10e-8:
##                eigVal[i] = np.sign(eigVal[i])*10e-8
#            coeff = np.dot(eigVec[:,i], gk)/(eigVal[i] + (np.sign(eigVal[i])*lambdaBar))
#            sbar -= coeff*eigVec[:,i]
#        return sbar
#    
#    def _lambda_bar(self, lambdaBar, trustR, eigVec, eigVal, gk):
#        sbar = self._sbar(eigVec, eigVal, gk, lambdaBar)
#        return np.dot(sbar, sbar) - trustR**2
#    
#    def _solve_sbar(self, trustR, gk, Hk, eigVec, eigVal):
#        lb = 0
#        ub = np.sqrt(np.trace(Hk**2)/Hk.shape[0])*10e10
#        stop = 10e-5
#
#        pos = [10e99]
#        neg = [-10e99]
#        for i in eigVal:
#            if i >= 0:
#                pos.append(i)
#            else:
#                neg.append(i)
#        
#        assert np.sign(self._lambda_bar(ub, trustR, eigVec, eigVal, gk)) < 0
#        lambdaBar = op.ridder(self._lambda_bar, lb, ub, (trustR, eigVec, eigVal, gk),stop)
#        result = self._sbar(eigVec, eigVal, gk, lambdaBar)
#        return result
#    
#    def _trust_radius(self, trustR, gk, gk1, Hk, sbar, boolAccept):
#        gk1Pred = gk + np.dot(Hk,sbar)
#        
#        normGk = np.linalg.norm(gk)
#        normGk1 = np.linalg.norm(gk1)
#        normGk1Pred = np.linalg.norm(gk1Pred)
#        
#        ro = self._mag_gradient(normGk, normGk1, normGk1Pred)
#        ki = self._cos_gradient(gk, gk1, gk1Pred)
#        d = gk.size
#        
##        print("ro", ro, "ki, ", ki, "d, ", d)
##        print("gk", normGk, "gk1", normGk1, "gk1Pred", normGk1Pred)
#        
#        if (4./5. < ro < 5./4.) and ki**2 > self._p10(d) and normGk1 < normGk:
#            step = sbar 
#            newTrustR = 2*trustR
#            print("3 success. new trustR, ", newTrustR)
#        elif (1./5. < ro < 5.) and ki**2 > self._p40(d) and normGk1 < normGk:
#            step = sbar
#            newTrustR = trustR
#            print("2 success. new trustR, ", newTrustR)
#        elif boolAccept:
#            newTrustR = trustR
#            step = sbar
#        elif normGk1 < normGk:
#            step = sbar
#            newTrustR = 0.75*trustR
#            print("1 success. new trustR, ", newTrustR)
#        else:
#            step = np.zeros_like(sbar)
#            newTrustR = 0.5*trustR
#            print("failed step. Reducing to ", newTrustR)
#
#        return step, newTrustR
    
    def _check_3rd_order(self,gk, gk1, Hk_inv, dx):
        return np.linalg.norm(dx - np.dot(Hk_inv,(gk1-gk)))
    
    def _unit_step(self, xk, dx, H, g):
        step = 1e-3
        x0 = cp.deepcopy(xk)
        x1 = cp.deepcopy(xk)
        
        for i in np.arange(3):
            for j in (0,1):
                x0[i][j]._array = np.zeros_like(xk[i][j]._array)
                x1[i][j]._array = np.ones_like(xk[i][j]._array)*step
                
        gk = g(*x0)
        gk1 = g(*x1)
        Hk = H(*x0)
        
        dx = np.ones_like(dx)*step
        
        return self._check_3rd_order(gk,gk1, Hk, dx)
    
    def _check_symmetry(self, x):
        for i in x:
            for j in i:
                if isinstance(j, DenseOneBody):
                    assert (np.abs(j._array - j._array.T) < 10e-8).all()
    
    def _check_hess_sign(self,Hk, offset):
#        print(np.linalg.eigh(Hk[offset[0]:offset[1],offset[0]:offset[1]])[0])
        assert np.min(np.linalg.eigh(Hk[offset[0]:offset[1],offset[0]:offset[1]])[0]) > 0
        assert np.min(np.linalg.eigh(Hk[offset[1]:offset[2],offset[1]:offset[2]])[0]) > 0
        assert np.max(np.linalg.eigh(Hk[offset[2]:offset[3],offset[2]:offset[3]])[0]) < 0
        assert np.max(np.linalg.eigh(Hk[offset[3]:offset[4],offset[3]:offset[4]])[0]) < 0
#        return
    
    def _shift_blockeig(self, Hk_block, eps):
        eigval,eigvec = np.linalg.eigh(Hk_block)
        
        Hk_block_result = np.zeros_like(Hk_block);
        
        if eps >= 0: #positive semi-def block
            for i,eigval_i in enumerate(eigval):
#                print(np.max((eigval_i)))
                Hk_block_result += np.max((eigval_i, eps))*np.ma.outer(eigvec[:,i], eigvec[:,i])
        else: #negative semi-def block
            for i,eigval_i in enumerate(eigval):
#                print(np.max((eigval_i)))
                Hk_block_result += np.min((eigval_i, eps))*np.ma.outer(eigvec[:,i], eigvec[:,i])
        
#        print(eps, np.linalg.eigh(Hk_block_result)[0])
        assert( (eps > 0 and np.min(np.linalg.eigh(Hk_block_result)[0]) > 0) or (eps < 0 and np.max(np.linalg.eigh(Hk_block_result)[0]) < 0) )  
        
        return Hk_block_result
    
    def _check_occ(self,x0, S):
        result = []
        for i in (0,1):
            result.append(np.linalg.eigh(np.dot(np.dot(lnag.matfuncs.sqrtm(S._array),x0[0][i]._array),lnag.matfuncs.sqrtm(S._array)))[0])
        
        return result
    
    def _check_idem(self, x0, S):
        result = []
        for i in (0,1):
            result.append(np.linalg.eigh(np.dot(np.dot(lnag.matfuncs.sqrtm(S._array),x0[1][i]._array),lnag.matfuncs.sqrtm(S._array)))[0])
        
        return result
        