import numpy as np
import copy as cp
import horton
import time
import pylab

#TODO: profile to figure out a quick way of evaluating this function.
class Lagrangian(object):
    def __init__(self,sys,ham, constraints, ifHess = False, isFrac = False, isRestricted = False):
        self.sys = sys
        self.ham = ham
        self.S = self.toNumpy(sys.get_overlap())
        self.constraints = constraints
        self.ifHess = ifHess
        self.isFrac = isFrac
        self.isRestricted = isRestricted
        
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
        
        self.shapes = base_args + cons_args
        
        self.offsets = [0]
        
        self.fock_alpha = self.sys.lf.create_one_body(self.shapes[0])
        self.fock_beta = self.sys.lf.create_one_body(self.shapes[1])
            
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
        for i in self.constraints[0] + self.constraints[1]:
            hasNext = i.next() or hasNext
        if self.spinConstraints is not None:
            for i in self.spinConstraints:
                hasNext = i.next() or hasNext
        
        return hasNext
        
    def toOneBody(self, *args):
        result = []
        for i in args:
            assert isinstance(i,np.ndarray)
            temp = self.sys.lf.create_one_body(i.shape[0])
            temp._array = i
            result.append(temp)
            
        if len(result) == 1:
            return result[0]
        return result
    
    def toNumpy(self, *args):
        result = []
        for i in args:
            assert isinstance(i, horton.DenseOneBody)
            result.append(i._array)
         
        if len(result) == 1:
            return result[0]   
        return result
    
    def fdiff_gradient(self, *args):
        h = 1e-5
        fn = self.lagrangian_spin_frac
        
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
        
        if self.isUT:
            anfn = self.sym_lin_grad_wrap
        else:
            anfn = self.lin_grad_wrap
        
        result = []

#        for i in np.arange(540, x.size):
        for i in np.arange(x.size):
            tmpFwd = cp.deepcopy(x)
            tmpBack = cp.deepcopy(x)
            
            tmpFwd[i] += h                
            tmpBack[i] -= h
            
            print "evaluating gradient: " + str(i)
            
#            fdan_norm = op.check_grad(self.lagrange_x, self.sym_lin_grad_wrap, tmpFwd)
#            assert fdan_norm < 1e-4, fdan_norm
            fd = self.fdiff_hess_grad_grad(tmpFwd)
            an = self.sym_lin_grad_wrap(tmpFwd)
            fdan = fd - an 
            fdan_norm = np.linalg.norm(fdan)
            assert fdan_norm < 1e-4, ("Mismatch in finite difference and analytic gradient", fdan_norm, fd, an, self.offsets)
#            
            an = (anfn(tmpFwd) - anfn(tmpBack))/ (np.float64(2)*h)
#            an = (self.fdiff_hess_grad_grad(tmpFwd) - self.fdiff_hess_grad_grad(tmpBack))/ (np.float64(2)*h)
            result.append(an)

        result = np.vstack(result)

        self.check_sym(result)
        return result
    
    def fdiff_hess_grad_grad(self, x, fn=None):
        h = 1e-5
        fn = fn or self.lagrange_x
        
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
        
        self.sys.wfn.invalidate()
        self.ham.invalidate()
        self.sys.wfn.update_dm("alpha", self.toOneBody(da)) #TODO: Fix for restricted
        self.sys.wfn.update_dm("beta", self.toOneBody(db))
        self.sys.wfn.update_dm("full", self.toOneBody(da+db))
    
        self.fock_alpha.reset()
        self.fock_beta.reset()
        
        self.ham.compute_fock(self.fock_alpha, self.fock_beta)
        
#        self.sys.wfn.invalidate() #Used for debugging occupations in callback
#        self.sys.wfn.update_exp(self.fock_alpha, self.fock_beta, self.sys.get_overlap(), self.toOneBody(da), self.toOneBody(db)) #Used for callback debugging
        
        alpha_args.append(self.toNumpy(self.fock_alpha))
        beta_args.append(self.toNumpy(self.fock_beta))

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
            
#         if not self.isRestricted:
#             pivot = len(fixed_terms)/2
#             a = fixed_terms[0:pivot]
#             b = fixed_terms[pivot:]    
#             result = [j for i in zip(a,b) for j in i]

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
    
    def lagrange_x(self, x):
        if self.isUT:
            args = self.UTvecToMat(x)
        else:
            args = self.vecToMat(x)
        
        result = self.lagrangian_spin_frac(*args)
#        test = self.lagrangian_spin_frac_old(*args)
#        assert result - test < 1e-10

        return result 
    
    def lagrangian_spin_frac(self, *args):
        args = self.symmetrize(*args)
        
        alpha_args = args[:self.nfixed_args/2]
        if not self.isRestricted:
            beta_args = args[self.nfixed_args/2:self.nfixed_args]
        else:
            beta_args = []
        
        S = self.S
        energy_spin = self.energy_spin
    
        da = alpha_args[0]
        db = beta_args[0]
    
        result = energy_spin(da, db)
        
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
    
    def energy_spin(self, Da, Db):
        self.sys.wfn.invalidate()
        self.ham.invalidate()
        self.sys.wfn.update_dm("alpha", self.toOneBody(Da))
        self.sys.wfn.update_dm("beta", self.toOneBody(Db))
        self.sys.wfn.update_dm("full", self.toOneBody(Da+Db))
#       self.sys.wfn.update_dm("spin", self.toOneBody(Da-Db))
        result = self.ham.compute_energy()
#       print "The energy is " + str(result)
        
        return result
    
    def test_occ(self, D):
        L = 0.5*np.ones_like(D)
        idx = np.diag_indices_from(L)
        L[idx] = 1
        
        P = np.dot(D.ravel(), self.S.ravel())
        Na = np.dot(P.ravel(), L.ravel())
        
        return Na
    
    def callback_system(self, x, dummy2):
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
        
        if self.isUT:
            args = self.UTvecToMat(x)
        else:
            args = self.vecToMat(x)
        D = [args[0], args[self.nfixed_args/2]]
            
        self.e_hist.append(self.energy_spin(*D))
            
        print "Energy is " + str(self.e_hist[-1])
            
        self.nIter+=1
    
    def UTmatToVec(self, *args):
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
    
    def UTvecToMat(self,x):
        self.tri_offsets() ##TESTING
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
    
    def matToVec(self, *args):
        result = []
        for i in args:
            result.append(i.ravel())
            
        x = np.hstack(result)
        return x
    
    def vecToMat(self,x):
        self.full_offsets() ##TESTING
        args = []
        for i in np.arange(len(self.offsets)-1):
            args.append(x[self.offsets[i]:self.offsets[i+1]].reshape([self.shapes[i], self.shapes[i]]))
#        assert (np.abs(fdiff_gradient(fn, *args) - gradient(*args)) < 1e-6).all(), (np.abs(fdiff_gradient(fn, *args) - gradient(*args)) < 1e-6, fdiff_gradient(fn, *args), gradient(*args))
        return args
    
    def check_sym(self, *args):
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
    
    def plot_mat(self, mat):
        pylab.matshow(mat)
        pylab.show()
    
    def symmetrize(self, *args):
        result = []
        for i in args:
            result.append(0.5*(i+i.T))
            
        return result
    
    def tri_offsets(self):
        self.offsets = [0]
        for n in self.shapes:
            self.offsets.append(int((n + 1)*n/2.))
        self.offsets = np.cumsum(self.offsets)
        
    def full_offsets(self):
        self.offsets = [0]
        for n in self.shapes:
            self.offsets.append(n**2)
        self.offsets = np.cumsum(self.offsets)
        
    def lin_grad_wrap(self,x):
        self.full_offsets()
        args = self.vecToMat(x)
        
#        print args
#        print "\n\n"
#        time.sleep(5)
        
        sym_args = self.symmetrize(*args)
        self.check_sym(*sym_args)      
        
        grad = self.calc_grad(*sym_args)
            
        self.check_sym(*grad)
#        grad[0:2] = self.symmetrize(*grad[0:2]) #average roundoff error in dLdD
        
        result = self.matToVec(*grad)
        
        return result
    
    def sym_lin_grad_wrap(self,x): 
        self.tri_offsets()
        args = self.UTvecToMat(x)
        
#        print args
#        print "\n\n"
#        time.sleep(5)
        
#        self.check_sym(*args)
        
        grad = self.calc_grad(*args)
            
        self.check_sym(*grad)
#        grad = self.symmetrize(*grad)
        
        result = self.UTmatToVec(*grad)
        
        return result
    
    def calc_occupations(self, x):
        args = self.UTvecToMat(x)
        
        for i in (args[0], args[1]):
            ds = np.dot(i,self.S)
            print np.diag(ds)
#            for c in self.constraints[key]:
#                ds = np.dot(i,self.S)
#                print np.trace(np.dot(ds,c.L))
        
        
