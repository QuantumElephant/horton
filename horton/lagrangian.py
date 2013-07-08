import numpy as np
import copy as cp
from horton.MatrixHelpers import *
import time

#TODO: profile to figure out a quick way of evaluating this function.
class Lagrangian(object):
    """A Lagrangian and gradient
    
    """
    def __init__(self,sys,ham, constraints, matHelper = None, isTriu = True, ifHess = False, isFrac = False, isRestricted = False):
        '''
            **Arguments:**
            sys
                An instance of horton.system containing information about basis and atomic coordinates
            ham
                An instance of horton.wfn with hamiltonian terms
            constraints
                A flat list of horton.Constraint objects. The order must match the order of arguments passed to the optimizer
            
            **Optional Arguments:**
            matHelper
                An instance of horton.MatrixHelpers
            isTriu
                Boolean: Whether matrices should be kept as upper triangular when vectorized. Defaults to True
            ifHess
                Boolean: Whether we should calculate the Hessian and condition number 
            isFrac
                Boolean: Whether fractional occupations are allowed. Must have P matrix in arguments if True
            isRestricted
                Boolean: Perform spin restricted calculation (testing)
            
        '''
        self.sys = sys
        self.ham = ham
        self.constraints = constraints
        self.ifHess = ifHess
        self.isFrac = isFrac
        self.isRestricted = isRestricted
        self.isTriu = isTriu
        
        assert not any(isinstance(el, list) for el in constraints)
        
        nbasis = sys.wfn.nbasis
        if isFrac:
            print "Fractional occupations enabled"
            base_args = [nbasis]*6
            self.nfixed_args = 6
        else:
            print "Fractional occupations disabled"
            base_args = [nbasis]*4
            self.nfixed_args = 4
        
        cons_args = [1] * len(constraints)
        
        if matHelper is None:
            shapes = base_args + cons_args
            if isTriu:
                self.matHelper = TriuMatrixHelpers(sys, shapes)
            else:
                self.matHelper = FullMatrixHelpers(sys, shapes)
        else:
            self.matHelper = matHelper
            
        self.S = self.matHelper.toNumpy(sys.get_overlap())
        
        self.fock_alpha = sys.lf.create_one_body(sys.wfn.nbasis)
        self.fock_beta = sys.lf.create_one_body(sys.wfn.nbasis)
        self.alwaysCalcHam = False
            
        #debugging
        self.e_hist = []
        self.occ_hist_a = []
        self.e_hist_a = []
        self.occ_hist_b = []
        self.e_hist_b = []
        self.nIter = 0
        self.logNextIter = False
        
    def nextStep(self):
        print "Stepping to next system"
        
        hasNext = False
        for i in self.constraints:
            hasNext = i.next() or hasNext
        return hasNext
    
    def fdiff_gradient(self, *args):
        h = 1e-5
        fn = self.lagrangian
        
        result = []
    
        for iKey,i in enumerate(args):
                intermediate_result = np.zeros([i.size])
    
                it = np.nditer(intermediate_result, flags=['multi_index'])
                while not it.finished:
                    dim2_index = np.unravel_index(it.multi_index[0], i.shape)
    
                    tmpFwd = list(cp.deepcopy(args))
                    tmpBack = list(cp.deepcopy(args))
    
                    tmpFwd[iKey][dim2_index] += h                
                    tmpBack[iKey][dim2_index] -= h
    
                    intermediate_result[it.multi_index] = (fn(*tmpFwd) - fn(*tmpBack)) / (np.float64(2)*h)
    
                    it.iternext()
                result.append(intermediate_result)
        result = np.hstack(result)
        
        return result
    
    def fdiff_hess_grad_x(self, x):
        h = 1e-5
        
        anfn = self.grad_wrap
        
        result = []

#        for i in np.arange(540, x.size):
        for i in np.arange(x.size):
            tmpFwd = cp.deepcopy(x)
            tmpBack = cp.deepcopy(x)
            
            tmpFwd[i] += h                
            tmpBack[i] -= h
            
            print "evaluating gradient: " + str(i)
            
#            fdan_norm = op.check_grad(self.lagrange_wrap, self.grad_wrap, tmpFwd)
#            assert fdan_norm < 1e-4, fdan_norm
#            fd = self.fdiff_hess_grad_grad(tmpFwd)
#            an = self.grad_wrap(tmpFwd)
#            fdan = fd - an 
#            fdan_norm = np.linalg.norm(fdan)
#            assert fdan_norm < 1e-4, ("Mismatch in finite difference and analytic gradient", fdan_norm, fd, an)
#            
            an = (anfn(tmpFwd) - anfn(tmpBack))/ (np.float64(2)*h)
#            an = (self.fdiff_hess_grad_grad(tmpFwd) - self.fdiff_hess_grad_grad(tmpBack))/ (np.float64(2)*h)
            result.append(an)

        result = np.vstack(result)

        self.matHelper.check_sym(result)
        return result
    
    def fdiff_hess_grad_grad(self, x, fn=None):
        h = 1e-5
        fn = fn or self.lagrange_wrap
        
        result = []
        
        for i in np.arange(x.size):        
            tmpFwd = cp.deepcopy(x)
            tmpBack = cp.deepcopy(x)
            
            tmpFwd[i] += h                
            tmpBack[i] -= h
            
            tmpFwdMat = tmpFwd
            tmpBackMat = tmpBack
            
            result.append((fn(tmpFwdMat) - fn(tmpBackMat)) / (np.float64(2)*h))

        result = np.hstack(result)
        
        #check sym on D
#        Da = result[0:49].reshape(7,7)
#        Db = result[49:98].reshape(7,7)
##        Ba = result[98:147].reshape(7,7)
#        result[0:49] = 0.5*(Da+Da.T).ravel()
#        result[49:98] = 0.5*(Db+Db.T).ravel()
#        self.check_sym(Da,Db)
        
        return result
    
    def calc_grad(self, *args): #move to kwargs eventually
#         print ("gradient: ", args)
        
        alpha_args = list(args[:self.nfixed_args/2])
        if not self.isRestricted:
            beta_args = list(args[self.nfixed_args/2:self.nfixed_args])
        else:
            beta_args = []

        da = alpha_args[0]
        db = beta_args[0]
        
        S = self.S
        
        if self.alwaysCalcHam:
            self.ham.invalidate()
        else:
            [self.ham.cache.invalidate(i) for i in ('op_coulomb', 'op_exchange_fock_alpha', 'op_exchange_fock_beta')]
        self.sys.wfn.invalidate()
        self.sys.wfn.update_dm("alpha", self.matHelper.toOneBody(da)) #TODO: Fix for restricted
        self.sys.wfn.update_dm("beta", self.matHelper.toOneBody(db))
        self.sys.wfn.update_dm("full", self.matHelper.toOneBody(da+db))
    
        self.fock_alpha.reset()
        self.fock_beta.reset()
        
        self.ham.compute_fock(self.fock_alpha, self.fock_beta)
        
#        self.sys.wfn.invalidate() #Used for debugging occupations in callback
#        self.sys.wfn.update_exp(self.fock_alpha, self.fock_beta, self.sys.get_overlap(), self.matHelper.toOneBody(da), self.matHelper.toOneBody(db)) #Used for callback debugging
        
        alpha_args.append(self.matHelper.toNumpy(self.fock_alpha))
        beta_args.append(self.matHelper.toNumpy(self.fock_beta))

        fixed_terms = []        
        for i in (alpha_args, beta_args):
            if len(i) == 0:
                continue
            
            if self.isFrac:
                [D,B,P] = i[:self.nfixed_args/2]
            else:
                [D,B] = i[:self.nfixed_args/2]
            fock = i[-1]

            dLdD = fock
#            print "fock", dLdD
            
            sbs = reduce(np.dot,[S,B,S])
            sdsbs = reduce(np.dot,[S,D,sbs])
            sbsds = reduce(np.dot,[sbs,D,S])
            
            dLdD -= 0.5*(sbs - sdsbs - sbsds + sbs.T - sdsbs.T - sbsds.T)
#            print "fock - outside", dLdD
            
            #dL/dB block
            sds = reduce(np.dot,[S,D,S])
            sdsds = reduce(np.dot,[S,D,S,D,S])
            
            if self.isFrac:
                pp = np.dot(P,P)
            else:
                pp = np.zeros_like(sds)
            dLdB = -0.5*(sds - sdsds - pp + sds.T - sdsds.T - pp.T)
#            print "dLdB", dLdB
            
            if self.isFrac:
                #dL/dP block
                PB = np.dot(P,B)
                BP = np.dot(B,P)
                dLdP = 0.5*(PB + BP + PB.T + BP.T) 
#                print "dLdP", dLdP
        
            dLdD = dLdD.squeeze()
            dLdB = dLdB.squeeze()

            fixed_terms.append(dLdD)
            fixed_terms.append(dLdB)
            if self.isFrac:
                dLdP = dLdP.squeeze()
                fixed_terms.append(dLdP)
            
        result = fixed_terms
        
        assert len(self.constraints) == len(args[self.nfixed_args:])
        
        for con,mul in zip(self.constraints, args[self.nfixed_args:]):
            if con.select == "alpha":
                result[0]+= con.D_gradient(da, mul)
                result.append(con.self_gradient(da))
            elif con.select == "beta":
                result[self.nfixed_args/2]+= con.D_gradient(db, mul)
                result.append(con.self_gradient(db))
            elif con.select == "add" or con.select == "diff":
                result[0]+= con.D_gradient(da, mul)
                result[self.nfixed_args/2]+= con.D_gradient(db, mul)
                if con.select == "add":
                    result.append(con.self_gradient(da) + con.self_gradient(db))
                else:
                    result.append(con.self_gradient(da) - con.self_gradient(db))
            else:
                raise ValueError('The select argument must be alpha or beta or add or diff')
        
        return result
    
    def lagrange_wrap(self, x):
        args = self.matHelper.vecToMat(x)
        result = self.lagrangian(*args)

        return result 
    
    def lagrangian(self, *args):
        args = self.matHelper.symmetrize(*args)
        
        alpha_args = args[:self.nfixed_args/2]
        if not self.isRestricted:
            beta_args = args[self.nfixed_args/2:self.nfixed_args]
        else:
            beta_args = []
        
        S = self.S
    
        da = alpha_args[0]
        db = beta_args[0]
    
        result = self.energy(da, db)
        
        for i in (alpha_args, beta_args):
            if len(i) == 0:
                continue
            if self.isFrac:
                [D,B,P] = i[:self.nfixed_args/2]
            else:
                [D,B] = i[:self.nfixed_args/2]
            
            pauli_test = reduce(np.dot,[S,D,S]) - reduce(np.dot,[S,D,S,D,S])
            if self.isFrac:
                pauli_test -= np.dot(P,P)
            pauli_test = np.dot(B,pauli_test)
            result -= np.trace(pauli_test)

        for con,mul in zip(self.constraints, args[self.nfixed_args:]):
            if con.select == "alpha":
                result += con.lagrange(da, mul)
            elif con.select == "beta":
                result += con.lagrange(db, mul)
            elif con.select == "add":
                result += con.lagrange(da, mul)
                result += con.lagrange(db, mul)
            elif con.select == "diff":
                result += con.lagrange(da, mul)
                result -= con.lagrange(db, mul)
            else:
                raise ValueError('The select argument must be alpha or beta or add or diff')
        return result
    
    def energy_wrap(self, x):
        args = self.matHelper.vecToMat(x)
        return self.energy(args[0],args[self.nfixed_args/2])
    
    def energy(self, Da, Db):
        self.sys.wfn.invalidate()
        self.ham.invalidate()
        self.sys.wfn.update_dm("alpha", self.matHelper.toOneBody(Da))
        self.sys.wfn.update_dm("beta", self.matHelper.toOneBody(Db))
        self.sys.wfn.update_dm("full", self.matHelper.toOneBody(Da+Db))
#       self.sys.wfn.update_dm("spin", self.matHelper.toOneBody(Da-Db))
        result = self.ham.compute_energy()
#       print "The energy is " + str(result)
        
        return result
    
    def callback_system(self, x, fx):
#        self.occ_hist_a.append(self.sys.wfn.exp_alpha.occupations) #Not ordered
##        self.e_hist_a.append(self.sys.wfn.exp_alpha.energies)
#        print "occ alpha:", self.occ_hist_a[-1]
##        print "e alpha:", self.e_hist_a[-1]
#        if isinstance(self.sys.wfn, horton.wfn.OpenShellWFN):
#            self.occ_hist_b.append(self.sys.wfn.exp_beta.occupations) #Not ordered
##            self.e_hist_b.append(self.sys.wfn.exp_beta.energies)
#            print "occ beta:", self.occ_hist_b[-1]
##            print "e beta:", self.e_hist_b[-1]
#            
        if self.logNextIter: #THIS GOES SECOND
            if self.ifHess:
                hess = self.fdiff_hess_grad_x(x)
                np.savetxt("jacobian"+str(self.nIter), hess)
                print "The condition number is {:0.3e}".format(np.linalg.cond(hess))
            
            self.logNextIter=False
            self.t2 = time.time()
            print "Iter {:d} took {:0.3e} s".format(self.nIter, self.t2-self.t1)

        if self.nIter==0 or self.nIter%1500==0 : #THIS GOES FIRST
            if self.ifHess:
                hess = self.fdiff_hess_grad_x(x)
                np.savetxt("jacobian"+str(self.nIter), hess)
                print "The condition number is {:0.3e}".format(np.linalg.cond(hess))

            self.logNextIter = True
            self.t1 = time.time()
        
        self.e_hist.append(self.energy_wrap(x))
            
        print "Energy is " + str(self.e_hist[-1])

        if np.linalg.norm(fx) < 1e-2:
            self.alwaysCalcHam = True
        
        self.nIter+=1
    
    def grad_wrap(self,x): 
        args = self.matHelper.vecToMat(x)
        
        sym_args = self.matHelper.symmetrize(*args)
        self.matHelper.check_sym(*sym_args)
        
        grad = self.calc_grad(*args)
            
        self.matHelper.check_sym(*grad)
        result = self.matHelper.matToVec(*grad)
        
        return result
    
    def calc_occupations(self, x):
        args = self.matHelper.vecToMat(x)
        
        for i in (args[0], args[self.nfixed_args/2]):
            ds = np.dot(i,self.S)
            print np.diag(ds)
#            for c in self.constraints[key]:
#                ds = np.dot(i,self.S)
#                print np.trace(np.dot(ds,c.L))
        
        
