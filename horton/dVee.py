import numpy as np
import copy as cp

class dVee(object):
    def __init__(self, J, K, lf):
        raise NotImplementedError
    def dD(self,dm):
        raise NotImplementedError
    def dDD(self,dm):
        raise NotImplementedError
    def dDaDb(self,dm):
        raise NotImplementedError


class HF_dVee(dVee):
    def __init__(self, integral, nbasis, lf):
        self.int4d = integral
        self.lf = lf
        self.nbasis = nbasis

    def Vee(self,D):
        J_alpha = self.lf.create_one_body(self.nbasis)
        J_beta = self.lf.create_one_body(self.nbasis)
        K_alpha = self.lf.create_one_body(self.nbasis)
        K_beta = self.lf.create_one_body(self.nbasis)

        self.int4d.apply_direct(D[0], J_alpha)
        self.int4d.apply_direct(D[1], J_beta)
        self.int4d.apply_exchange(D[0], K_alpha)
        self.int4d.apply_exchange(D[1], K_beta)

        Jaa = J_alpha.expectation_value(D[0])
        Jab = J_alpha.expectation_value(D[1])
        Jbb = J_beta.expectation_value(D[1])
        
        Kaa = K_alpha.expectation_value(D[0])
        Kbb = K_beta.expectation_value(D[1])

        result = 0.5*(Jaa + 2*Jab + Jbb - Kaa - Kbb)

        return result

    def dD(self,D):
        J2_alpha = self.lf.create_one_body(self.nbasis)
        K2_alpha = self.lf.create_one_body(self.nbasis)
        J2_beta = self.lf.create_one_body(self.nbasis)

        self.int4d.apply_direct(D[0], J2_alpha)
        self.int4d.apply_exchange(D[0], K2_alpha)
        self.int4d.apply_direct(D[1], J2_beta)

        result = self.lf.create_one_body(self.nbasis) #TODO inefficient
        result._array = (J2_alpha._array + J2_beta._array - K2_alpha._array)

        return result

    def dD2(self,dm):
        J_K = self.lf.create_two_body(self.nbasis)
        J_K._array = cp.deepcopy(np.swapaxes(self.int4d._array, 1,2) - np.swapaxes(self.int4d._array, 2,3))
#        J_K._array = cp.deepcopy(np.swapaxes(self.int4d._array, 1,2) - self.int4d._array)
#        J_K._array = self.int4d._array - np.swapaxes(self.int4d._array, 1,2)

        return J_K

    def dDaDb(self, dm):
        J = self.lf.create_two_body(self.nbasis)
        J._array = cp.deepcopy(np.swapaxes(self.int4d._array, 1,2))
#        J = cp.deepcopy(self.int4d)
        return J