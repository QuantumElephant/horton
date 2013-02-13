import numpy as np
import copy as cp
import scipy.linalg.matfuncs as mat
import horton
import time
import pylab
import scipy.optimize as op

#TODO: profile to figure out a quick way of evaluating this function.
class lagrangian(object):
    def __init__(self, lf, S, N, N2, matrix_args, sys, ham):
        self.lf = lf
        self.S = self.toNumpy(S)
        self.N = N
        self.N2 = N2

        self.shapes = []
        
        self.matrix_args = matrix_args
        
        self.offsets = [0]
        
        for i in matrix_args:
            self.shapes.append(i.shape[0])
        
        self.sys = sys
        self.ham = ham
        
        self.fock_alpha = self.lf.create_one_body(self.shapes[0])
        self.fock_beta = self.lf.create_one_body(self.shapes[1])
            
        #debugging
        self.e_hist = []
        self.occ_hist_a = []
        self.e_hist_a = []
        self.occ_hist_b = []
        self.e_hist_b = []
        self.nIter = 0
        self.logNextIter = False
        
    def toOneBody(self, *args):
        result = []
        for i in args:
            assert isinstance(i,np.ndarray)
            temp = self.lf.create_one_body(i.shape[0])
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
        fn = self.fdiff_hess_grad_grad
        
        if self.isUT:
            anfn = self.sym_lin_grad_wrap
        else:
            anfn = self.lin_grad_wrap
        
        result = []

        for i in np.arange(x.size):
            tmpFwd = cp.deepcopy(x)
            tmpBack = cp.deepcopy(x)
            
            tmpFwd[i] += h                
            tmpBack[i] -= h
            
            print "evaluating gradient"
            
#            fdan = op.check_grad(self.lagrange_x, self.sym_lin_grad_wrap, tmpFwd)
#            assert fdan < 1e-4, fdan 
            
            an = (anfn(tmpFwd) - anfn(tmpBack))/ (np.float64(2)*h)
            result.append(an)

        result = np.vstack(result)
        self.check_sym(result)
        return result
    
    def fdiff_hess_grad_grad(self, x):
        h = 1e-5
        fn = self.lagrangian_spin_frac
        
        result = []
        
        if self.isUT:
            reshapeFn = self.UTvecToMat
        else:
            reshapeFn = self.vecToMat
        
        for i in np.arange(x.size):        
            tmpFwd = cp.deepcopy(x)
            tmpBack = cp.deepcopy(x)
            
            tmpFwd[i] += h                
            tmpBack[i] -= h
            
            tmpFwdMat = reshapeFn(tmpFwd)
            tmpBackMat = reshapeFn(tmpBack)
            
            result.append((fn(*tmpFwdMat) - fn(*tmpBackMat)) / (np.float64(2)*h))

        result = np.hstack(result)
        return result
    
    def calc_grad(self, *args):
        [da, db, ba, bb, pa, pb, mua, mub] = args
        S = self.S
        N = self.N
        N2 = self.N2
        toNumpy = self.toNumpy
        
        result = []
    
        self.sys.wfn.invalidate()
        self.ham.invalidate()
        self.sys.wfn.update_dm("alpha", self.toOneBody(da))
        self.sys.wfn.update_dm("beta", self.toOneBody(db))
        self.sys.wfn.update_dm("full", self.toOneBody(da+db))
    
        self.fock_alpha.reset()
        self.fock_beta.reset()
        
        self.ham.compute_fock(self.fock_alpha, self.fock_beta)
        
        self.sys.wfn.invalidate()
        self.sys.wfn.update_exp(self.fock_alpha, self.fock_beta, self.sys.get_overlap(), self.toOneBody(da), self.toOneBody(db))
        
        numpy_fock_alpha = toNumpy(self.fock_alpha)
        numpy_fock_beta = toNumpy(self.fock_beta)
#        
        for spins in [[numpy_fock_alpha, da,ba,pa,mua,N], [numpy_fock_beta, db,bb,pb,mub,N2]]:
            [fock,D,B,P,Mu,n] = spins
            dLdD = fock
            
            sbs = np.dot(np.dot(S,B),S)
            sdsbs = np.dot(np.dot(S,D),sbs)
            sbsds = np.dot(np.dot(sbs,D),S)    
            outside = sbs - sdsbs - sbsds + sbs.T - sdsbs.T - sbsds.T
            dLdD -= 0.5*outside
        
            dLdD -= Mu*S
        
            #dL/dB block
            sds = np.dot(np.dot(S,D),S)
            sdsds = np.dot(np.dot(np.dot(np.dot(S,D),S),D),S)
            pp = np.dot(P,P)
            dLdB = -0.5*(sds - sdsds - pp + sds.T - sdsds.T - pp.T)
            
            #dL/dP block
            PB = np.dot(P,B)
            BP = np.dot(B,P)
            dLdP = 0.5*(PB + BP + PB.T + BP.T) 
        
            #dL/d_mu block
            dLdMu = n - np.trace(np.dot(S,D))
            
            dLdD = dLdD.squeeze()
            dLdB = dLdB.squeeze()
            dLdP = dLdP.squeeze()
            
            result.append(dLdD)
            result.append(dLdB)
            result.append(dLdP)
            result.append(dLdMu)
            
        a = result[0:4]
        b = result[4:8]    
        c = [j for i in zip(a,b) for j in i]    
        return c  
    
    def lagrange_x(self, x):
        args = self.UTvecToMat(x)
        return self.lagrangian_spin_frac(*args)
    
    def lagrangian_spin_frac(self, Da, Db, Ba, Bb, Pa, Pb, Mua, Mub):
        S = self.S
        N = self.N
        N2 = self.N2
        energy_spin = self.energy_spin
        
        Da = 0.5*(Da+Da.T)
        Db = 0.5*(Db+Db.T)
        Ba = 0.5*(Ba+Ba.T)
        Bb = 0.5*(Bb+Bb.T)
        Pa = 0.5*(Pa+Pa.T)
        Pb = 0.5*(Pb+Pb.T)
    
        result = 0
        result += energy_spin(Da, Db)
        
        for spin in [[Da,Ba,Pa,Mua,N], [Db,Bb,Pb,Mub,N2]]:
            [D,B,P,Mu,n] = spin
            pauli_test = np.dot(B,np.dot(np.dot(S,D),S) - np.dot(np.dot(np.dot(np.dot(S,D),S),D),S) - np.dot(P,P))
            result -= np.trace(pauli_test)
            result -= np.squeeze(Mu*(np.trace(np.dot(D,S)) - n))
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
#        self.occ_hist_a.append(self.sys.wfn.exp_alpha.occupations)
#        self.e_hist_a.append(self.sys.wfn.exp_alpha.energies)
#        print "occ alpha:", self.occ_hist_a[-1]
#        print "e alpha:", self.e_hist_a[-1]
#        if isinstance(self.sys.wfn, horton.wfn.OpenShellWFN):
#            self.occ_hist_b.append(self.sys.wfn.exp_beta.occupations)
#            self.e_hist_b.append(self.sys.wfn.exp_beta.energies)
#            print "occ beta:", self.occ_hist_b[-1]
#            print "e beta:", self.e_hist_b[-1]
#            
        if self.logNextIter: #THIS GOES SECOND
#            hess = self.fdiff_hess_grad_x(x)
#            np.savetxt("jacobian"+str(self.nIter), hess)
#            print "The condition number is {:0.3e}".format(np.linalg.cond(hess))
            
            self.logNextIter=False
            self.t2 = time.time()
            print "Iter {:d} took {:0.3e} s".format(self.nIter, self.t2-self.t1)

        if self.nIter==0 or self.nIter%1500==0 : #THIS GOES FIRST
#            hess = self.fdiff_hess_grad_x(x)
#            np.savetxt("jacobian"+str(self.nIter), hess)
#            print "The condition number is {:0.3e}".format(np.linalg.cond(hess))
#            self.check_sym(hess)


            self.logNextIter = True
            self.t1 = time.time()
        
        if self.isUT:
            D = self.UTvecToMat(x)[:2]
        else:
            D = self.vecToMat(x)[:2]
            
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
            symerror = np.abs(i[:,:shortDim] - i.T[:shortDim,:]) 
            assert (symerror < 1e-8).all(), (np.vstack(np.where(symerror > 1e-8)).T, symerror, self.plot_mat(symerror > 1e-8))
    
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
        for i in self.matrix_args:
            n = i.shape[0]
            self.offsets.append(int((n + 1)*n/2.))
        self.offsets = np.cumsum(self.offsets)
        
    def full_offsets(self):
        self.offsets = [0]
        for i in self.matrix_args:
            self.offsets.append(i.size)
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