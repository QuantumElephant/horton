import numpy as np
import copy as cp
import scipy.linalg.matfuncs as mat
import horton
import time
import pylab
import scipy.optimize as op

#TODO: profile to figure out a quick way of evaluating this function.
class lagrangian(object):
    def __init__(self, lf, T, V, S, N, N2, dVee, W, matrix_args, sys = None, ham = None):
        [T, V, S] = self.toNumpy(T,V,S)
        
        self.lf = lf
        self.T = T
        self.V = V #TODO: check to see if potentials are spin restricted
        self.S = S
        self.N = N
        self.N2 = N2
        self.W = W
        self.fn = self.lagrangian_spin_frac
        self.grad = self.gradient_spin_frac
#        self.grad = self.fdiff_gradient

        self.dVee = dVee
        
        self.shapes = []
        
        self.matrix_args = matrix_args
        
        self.offsets = [0]
        
        for i in matrix_args:
#            self.offsets.append(i.size)
            self.shapes.append(i.shape[0])
##            self.offsets.append(np.triu_indices(self.shapes[-1])[0].size)
        
#        self.offsets = np.cumsum(self.offsets)
        
        self.sys = sys
        self.ham = ham
        
        self.fock_alpha = self.lf.create_one_body(self.shapes[0])
        self.fock_beta = self.lf.create_one_body(self.shapes[1])
#        
#        self.sys.wfn.update_dm("alpha", self.toOneBody(self.lf,matrix_args[0])[0])
#        self.sys.wfn.update_dm("beta", self.toOneBody(self.lf,matrix_args[1])[0])
#        
#        if self.ham.grid is None:
#            self.ham.compute_fock(self.fock_alpha, None)
#            self.ham.compute_fock(self.fock_beta, None) #print "HACKHACKHACK! THIS IS FOR HF!!!!!!"
#        else:
#            self.ham.compute_fock(self.fock_alpha, self.fock_beta) #print "HACK HACK HACK! THIS IS FOR DFT!!!!"
            
            
        #debugging
        self.e_hist = []
        self.occ_hist_a = []
        self.e_hist_a = []
        self.occ_hist_b = []
        self.e_hist_b = []
        self.nIter = 0
        self.logNextIter = False
        
    def toOneBody(self, lf, *args):
        result = []
        assert isinstance(lf, horton.LinalgFactory)
        for i in args:
            assert isinstance(i,np.ndarray)
            temp = lf.create_one_body(i.shape[0])
            temp._array = i
            result.append(temp)
        return result
    
    def toNumpy(self, *args):
        result = []
        for i in args:
            assert isinstance(i, horton.DenseOneBody)
            result.append(i._array)
        return result
    
    def fdiff_gradient(self, *args):
        h = 1e-5
        fn = self.fn
        
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
    
    def fdiff_hess_slow(self, fn, *args):
        h = 1e-5
        
        result = []
        
        for key, i in enumerate(args):
            it = np.nditer(i, flags=["multi_index"])
            while not it.finished:
#                dim2_index = np.unravel_index(it.multi_index[0], i.shape)
                
                tmpFwd = list(cp.deepcopy(args))
                tmpBack = list(cp.deepcopy(args))
                
                tmpFwd[key][it.multi_index] += h                
                tmpBack[key][it.multi_index] -= h

                print "evaluating gradient"
                result.append((fn(*tmpFwd) - fn(*tmpBack)) / (np.float64(2)*h))

                it.iternext()
                
        result = np.vstack(result)
        return result
    
    def fdiff_hess_slow_x(self, fn, x):
        h = 1e-5
        
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

            print "evaluating gradient"
            result.append((fn(*tmpFwdMat) - fn(*tmpBackMat)) / (np.float64(2)*h))

        result = np.vstack(result)
        return result
    
    def fdiff_hess_grad_x(self, x):
        hOrig = 1e-5
        fn = self.fdiff_hess_grad_grad
        
        if self.isUT:
            anfn = self.sym_lin_grad_wrap
#            UTsize = self.offsets[1] #hack! assumes same size of basis for all lagrange multipliers at front of args
#            UTdiags = self.gen_UT_index(self.shapes[0])
            
        else:
            anfn = self.lin_grad_wrap
            h = hOrig
        h = hOrig
        
        result = []
#        result2 = []

        for i in np.arange(x.size):
#            if self.isUT:
#                matInd = i%UTsize
#                if matInd not in UTdiags:
#                    h = hOrig/2.
#                else:
#                    h = hOrig
             
            tmpFwd = cp.deepcopy(x)
            tmpBack = cp.deepcopy(x)
            
            tmpFwd[i] += h                
            tmpBack[i] -= h
            
            print "evaluating gradient"
            
#            fdan = np.abs(fn(tmpFwd) - anfn(tmpFwd))
#            assert (fdan < 1e-6).all(), (fdan > 1e-6, np.where(fdan > 1e-6), fdan)

#            fdan = op.check_grad(self.lagrange_x, self.sym_lin_grad_wrap, tmpFwd)
#            assert fdan < 1e-4, fdan 
            
#            fd = (fn(tmpFwd) - fn(tmpBack))/(np.float64(2)*h)
            an = (anfn(tmpFwd) - anfn(tmpBack))/ (np.float64(2)*h)
            
            result.append(an)
#            result2.append(fd)

        result = np.vstack(result)
#        result2 = np.vstack(result2)
        
#        self.plot_mat((result - result2) > 1e-6)
        
        self.check_sym(result)
        return result
    
    def gen_UT_index(self, ndim):
        result = [0]
        last = 1
        for i in np.arange(ndim):
            result.append(result[-1] + last)
            last += 1
            
        result = result[::-1]
        result = np.array(result)
        result = np.abs(result - result[0]) 
        result = result[:-1]
        
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
#        offsets = self.offsets
        S = self.S
        N = self.N
        N2 = self.N2
        toNumpy = self.toNumpy
        toOneBody = self.toOneBody
        
        result = []
    
        self.sys.wfn.invalidate()
        self.ham.invalidate()
        self.sys.wfn.update_dm("alpha", self.toOneBody(self.lf,da)[0])
        self.sys.wfn.update_dm("beta", self.toOneBody(self.lf,db)[0])
        self.sys.wfn.update_dm("full", self.toOneBody(self.lf,(da+db))[0])
#            self.sys.wfn.update_dm("spin", self.toOneBody(self.lf,(da-db))[0])
    
        self.fock_alpha.reset()
        self.fock_beta.reset()
        
        if self.ham.grid is None:
            self.ham.compute_fock(self.fock_alpha, self.fock_beta) #print "HACK HACK HACK! THIS IS FOR DFT!!!!"
#
#            self.ham.compute_fock(self.fock_alpha, None)
#            self.ham.compute_fock(self.fock_beta, None) #print "HACKHACKHACK! THIS IS FOR HF!!!!!!"
        else:
            self.ham.compute_fock(self.fock_alpha, self.fock_beta) #print "HACK HACK HACK! THIS IS FOR DFT!!!!"
        
        self.sys.wfn.invalidate()
        self.sys.wfn.update_exp(self.fock_alpha, self.fock_beta, self.sys.get_overlap(), toOneBody(self.lf,da)[0], toOneBody(self.lf,db)[0])
        
        numpy_fock_alpha = toNumpy(self.fock_alpha)
        numpy_fock_beta = toNumpy(self.fock_beta)
#        
        for spins in [[numpy_fock_alpha, da,ba,pa,mua,N], [numpy_fock_beta, db,bb,pb,mub,N2]]:
#        for spins in [[da,ba,pa,mua,N], [db,bb,pb,mub,N2]]:
#            dLdD = self.old_grad_fock(da,db)
            
#            [D,B,P,Mu,n] = spins
#            dLdD = 0
            [fock,D,B,P,Mu,n] = spins
            dLdD = fock[0]
            
#            wrappedD = toOneBody(self.lf, da, db, D)
#            dLdD = toNumpy(self.dVee.dD(*wrappedD))
#            dLdD += (self.T - self.V)
            
#            assert (np.abs(fock - self.old_grad_fock(da, db)) < 1e-5).all(), fock-self.old_grad_fock(da, db)
            
            
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
    
    def gradient_spin_frac(self, *args):
        """Receives matrices from vecToMat
        """
        sym_args = args          
        sym_args = self.symmetrize(*args)
        self.check_sym(*sym_args)      
        
        grad = self.calc_grad(*sym_args)
            
        self.check_sym(*grad)
        grad[0:2] = self.symmetrize(*grad[0:2]) #average roundoff error in dLdD
        
        result = self.matToVec(*grad)
            
#        self.check_UT(grad, *args)
        return result
    
    
    def sym_gradient_spin_frac(self, *args):
        """Receives full matrices from UTVecToMat
        """
        self.check_sym(*args)
        
        grad = self.calc_grad(*args)
            
        self.check_sym(*grad)
#        grad = self.symmetrize(*grad)
        
        result = self.UTmatToVec(*grad)
        
        return result
    
    def old_grad_fock(self, da, db):
        T = self.T
        V = self.V
        dVee = self.dVee
        toNumpy = self.toNumpy
        toOneBody = self.toOneBody
        
        wrappedD = toOneBody(self.lf, da, db)
        dLdD = toNumpy(dVee.dD(wrappedD))
        dLdD += (T - V)
        
        return dLdD
    
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
#        if self.isUT:
#            Da -= 0.5*np.diag(np.diag(Da))
#            Db -= 0.5*np.diag(np.diag(Db))
        
        
        if self.sys is None or self.ham is None:
            result = self.old_energy_spin(Da, Db)
        else:
            self.sys.wfn.invalidate()
            self.ham.invalidate()
            self.sys.wfn.update_dm("alpha", self.toOneBody(self.lf,Da)[0])
            self.sys.wfn.update_dm("beta", self.toOneBody(self.lf,Db)[0])
            self.sys.wfn.update_dm("full", self.toOneBody(self.lf,(Da+Db))[0])
#            self.sys.wfn.update_dm("spin", self.toOneBody(self.lf,(Da-Db))[0])
            result = self.ham.compute_energy()
#            print "The energy is " + str(result)
#            self.check_wfn_energy(Da,Db,result)        
        
        return result
    
    def old_energy_spin(self,Da,Db):
        T = self.T
        V = self.V
        dVee = self.dVee
        toNumpy = self.toNumpy
        toOneBody = self.toOneBody
        
        wrappedD = toOneBody(self.lf, Da, Db)
        result = dVee.Vee(wrappedD)
        result += np.trace(np.dot((T-V),Da))
        result += np.trace(np.dot((T-V),Db))
        return result
    
    def check_wfn_energy(self, Da, Db, trial):
        result = self.old_energy_spin(Da, Db)
        
#        print "Horton", trial
#        print "Own", result
#        print np.abs(result - trial)
        
        assert np.abs(trial - result) < 1e-4, np.abs(trial - result)

    def check_occ(self,x0):
        S = self.S
        result = np.linalg.eigh(np.dot(np.dot(mat.sqrtm(S),x0[0]),mat.sqrtm(S)))[0]
        if len(x0)==1:
            return result
        result2 = np.linalg.eigh(np.dot(np.dot(mat.sqrtm(S),x0[1]),mat.sqrtm(S)))[0]
        return result,result2

    def wrap_callback_spin(self,x,f=None):
        energy_spin = self.energy_spin
        offsets = self.offsets
        check_occ = self.check_occ
        S = self.S
        
        Da = x[offsets[0]:offsets[1]].reshape([self.shapes[0],self.shapes[0]])
        Db = x[offsets[1]:offsets[2]].reshape([self.shapes[1], self.shapes[1]])
        
        print("energy", energy_spin(Da,Db))
        print("normalization",np.trace(np.dot(Da,S)),np.trace(np.dot(Db,S)), check_occ([Da,Db]))
        print("\n")
        
        return energy_spin(Da,Db)
    
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
#            if self.isUT:
#                hess = self.fdiff_hess_slow_x(self.sym_gradient_spin_frac, x)
#            else:
#                hess = self.fdiff_hess_slow_x(self.gradient_spin_frac,x)
#            np.savetxt("jacobian"+str(self.nIter), hess)
#            print "The condition number is {:0.3e}".format(np.linalg.cond(hess))
            
            self.logNextIter=False
            self.t2 = time.time()
            print "Iter {:d} took {:0.3e} s".format(self.nIter, self.t2-self.t1)

        if self.nIter==0 or self.nIter%1500==0 : #THIS GOES FIRST
            if self.isUT:
#                hess = self.fdiff_hess_slow_x(self.sym_gradient_spin_frac, x)
                hess = self.fdiff_hess_grad_x(x)
            else:
#                hess = self.fdiff_hess_slow_x(self.gradient_spin_frac, x)
                hess = self.fdiff_hess_grad_x(x)
                
            np.savetxt("jacobian"+str(self.nIter), hess)
            print "The condition number is {:0.3e}".format(np.linalg.cond(hess))
            self.check_sym(hess)


            self.logNextIter = True
            self.t1 = time.time()
        
        if self.isUT:
            D = self.UTvecToMat(x)[:2]
        else:
            D = self.vecToMat(x)[:2]
            
        self.e_hist.append(self.energy_spin(*D))
            
        print "Energy is " + str(self.e_hist[-1])
            
        self.nIter+=1
        
    def check_UT(self, full_grad, *args):
        x = self.matToVec(*args)
        
        sym_grad = self.sym_lin_grad_wrap(x)
        assert np.linalg.norm(full_grad - sym_grad) < 1e-10, np.abs(full_grad - sym_grad)
        
    
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
    
    def matToVec(self, *args):
        result = []
        for i in args:
            result.append(i.ravel())
            
        x = np.hstack(result)
        return x
    
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
    
    def vecToMat(self,x):
        self.full_offsets() ##TESTING
        args = []
        for i in np.arange(len(self.offsets)-1):
            args.append(x[self.offsets[i]:self.offsets[i+1]].reshape([self.shapes[i], self.shapes[i]]))
#        assert (np.abs(fdiff_gradient(fn, *args) - gradient(*args)) < 1e-6).all(), (np.abs(fdiff_gradient(fn, *args) - gradient(*args)) < 1e-6, fdiff_gradient(fn, *args), gradient(*args))
        return args
    
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
#            temp = ut + np.triu(ut, 1).T
            temp = 0.5*(ut + ut.T)
            args.append(temp)
        self.check_sym(*args)
        return args
    
    def test_UTconvert(self,x):
        xOrig = cp.deepcopy(x)
        
        a = self.UTvecToMat(x)
        b = self.UTmatToVec(*a)
        
        assert (np.abs(b - xOrig) < 1e-10).all()
    
    def tri_offsets(self):
        self.offsets = [0]
        for i in self.matrix_args:
#            self.offsets.append(np.triu_indices(self.shapes[key])[0].size)
            n = i.shape[0]
            self.offsets.append(int((n + 1)*n/2.))
        self.offsets = np.cumsum(self.offsets)
        
    def full_offsets(self):
        self.offsets = [0]
        for i in self.matrix_args:
            self.offsets.append(i.size)
        self.offsets = np.cumsum(self.offsets)
        
    def check_fd(self, an, *args):
        fdan = np.abs(self.fdiff_gradient(*args) - an)
        assert (fdan < 1e-6).all(), (fdan < 1e-6, np.where(fdan > 1e-6), fdan)
        
    def check_UT_fd(self, an, *args):
        full_fd = self.fdiff_gradient(*args)
        fd = self.vecToMat(full_fd)
        self.check_sym(*fd)
        fd = self.UTmatToVec(*fd)
        
#        print np.linalg.norm(fd-an)
        
        fdan = np.abs(fd - an)
        assert (fdan < 1e-6).all(), (fdan > 1e-6, np.where(fdan > 1e-6), fdan)
    
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
        
#        self.check_fd(result, *args)

        
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

        
#        self.check_UT_fd(result, *args)
        

        
        return result