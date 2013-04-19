from horton.newton_rewrite import NewtonKrylov
from horton import *
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import copy as cp

np.set_printoptions(threshold = 2000)

def sqrtm(A):
    result = np.zeros_like(A)
    eigvs,eigvc = np.linalg.eigh(A)
    for i in np.arange(len(eigvs)):
        if eigvs[i] > 0:
            result += np.sqrt(eigvs[i])*np.outer(eigvc[i], eigvc[i])
    
    return result

def promol_orbitals(sys, basis, numChargeMult=None, ifCheat = False):
    orb_alpha = []
    orb_beta = []
    occ_alpha = []
    occ_beta = []
    e_alpha = []
    e_beta = []
    nbasis = []
    
    for atomNum,i in enumerate(sys.numbers):
        atom_sys = System(np.zeros((1,3), float), np.array([i]), obasis=basis) #hacky
        
        if isinstance(numChargeMult, np.ndarray) and atomNum in numChargeMult[:,0]:
            [num, charge, mult] = numChargeMult[np.where(numChargeMult[:,0] == atomNum),:].squeeze()
            print "REWRITING atom number " + str(num) + " element: " + str(i) + " with charge: " \
                + str(charge) + " and multiplicity: " + str(mult)
            
            atom_sys.init_wfn(charge=charge, mult=mult,restricted=False)
        else:
            atom_sys.init_wfn(restricted=False)        

        nbasis.append(atom_sys.wfn.nbasis)
        guess_hamiltonian_core(atom_sys)

#DFT
#        grid = BeckeMolGrid(system, random_rotate=False)
#        
#        libxc_term = LibXCLDATerm('x')
#        ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

#HF
        ham = Hamiltonian(atom_sys, [HartreeFock()])
        
        converge_scf_oda(ham)
        
        orb_alpha.append(atom_sys.wfn.exp_alpha._coeffs)
        orb_beta.append(atom_sys.wfn.exp_beta._coeffs)
        
        occ_alpha.append(atom_sys.wfn.exp_alpha.occupations)
        occ_beta.append(atom_sys.wfn.exp_beta.occupations)
        
        e_alpha.append(atom_sys.wfn.exp_alpha.energies)
        e_beta.append(atom_sys.wfn.exp_beta.energies)
        
    orb_alpha = scp.linalg.block_diag(*orb_alpha)
    orb_beta = scp.linalg.block_diag(*orb_beta)
    
    occ_alpha = np.hstack(occ_alpha)
    occ_beta = np.hstack(occ_beta)

    e_alpha = np.hstack(e_alpha)
    e_beta = np.hstack(e_beta)
    
    if ifCheat:
        print "CHEAT MODE ENABLED"
        cheatSys = calc_DM(sys)
        
        orb_alpha = cheatSys.wfn.exp_alpha._coeffs
        orb_beta = cheatSys.wfn.exp_beta._coeffs
        
        occ_alpha = cheatSys.wfn.exp_alpha.occupations
        occ_beta = cheatSys.wfn.exp_beta.occupations
        
        e_alpha = cheatSys.wfn.exp_alpha.energies
        e_beta = cheatSys.wfn.exp_beta.energies
        
    return orb_alpha, orb_beta, occ_alpha, occ_beta, e_alpha, e_beta, nbasis
    
def promol_guess(orb_a, orb_b, occ_a, occ_b, energy_a, energy_b, N = None, N2 = None):
#    occ_a = #TODO: add 6-31++G** occupations 

    assert occ_a.size == orb_a.shape[0] and occ_b.size == orb_b.shape[0], "initial occupation size: " + str(occ_a.size) + "," + str(occ_b.size) + " basis size: " + str(orb_a.shape[0]) + "," + str(orb_b.shape[0])

    if N is None or N2 is None:
        N = np.sum(occ_a)
        N2 = np.sum(occ_b)
    
    pro_da = np.zeros_like(orb_a)
    pro_ba = np.zeros_like(orb_a)
    pro_db = np.zeros_like(orb_a)
    pro_bb = np.zeros_like(orb_a)
    
    for i in np.arange(orb_a.shape[0]):
        psi_psi_a = np.outer(orb_a[:,i], orb_a[:,i])
        pro_da += psi_psi_a*occ_a[i]
        pro_ba += psi_psi_a*np.abs(energy_a[i])
        
        psi_psi_b = np.outer(orb_b[:,i], orb_b[:,i])
        pro_db += psi_psi_b*occ_b[i]
        pro_bb += psi_psi_b*np.abs(energy_b[i])
    
    mua = np.ones([1])*np.max(energy_a[energy_a<0]) #hack. Can't handle size 1 arrays cleanly.
    mub = np.ones([1])*np.max(energy_a[energy_a<0])
    
    return pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2

def calc_DM(sys):
    system = System(sys.coordinates, sys.numbers, obasis=sys.obasis)
    system.init_wfn(restricted=False) 
    guess_hamiltonian_core(system)
#  DFT  
    grid = BeckeMolGrid(system, random_rotate=False)
    libxc_term = LibXCLDATerm('x')
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
    
#  HF
#    ham = Hamiltonian(system, [HartreeFock()])
    
    converged = converge_scf_oda(ham, max_iter=5000)
    return system

def promol_frac(sys, pro_da, pro_db):
    S = sys.get_overlap()._array
    pa = sqrtm(reduce(np.dot,[S,pro_da,S]) - reduce(np.dot,[S,pro_da,S,pro_da,S]))
    pb = sqrtm(reduce(np.dot,[S,pro_db,S]) - reduce(np.dot,[S,pro_db,S,pro_db,S]))
    
    return pa, pb
    
def prep_D(*args):
    result = []
    for i in args:
        if i.size == 1:
            result.append(i.squeeze())
            continue
        diag_idx = np.diag_indices_from(i)
        ut_idx = np.triu_indices_from(i)
        i[diag_idx] *= 0.5
        result.append(2*i[ut_idx])
    return np.hstack(result)

# def project(origSys, projectedBasis, *args):
#     projectedSys = System(origSys.coordinates, origSys.numbers, obasis=projectedBasis)
#      
#     sOrig = origSys.get_overlap()._array
#     sProj = projectedSys.get_overlap()._array
#      
#     sOrigProj = sProj[:sOrig.shape[0], sOrig.shape[1]:]
#     sProjInv = np.linalg.inv(sProj[sOrig.shape[0]:, sOrig.shape[1]:])
#     
#     result = []
#     for i in args:
#         result.append(reduce(np.dot,[sProjInv,sOrigProj.T,i,sOrigProj,sProjInv]))
#     

def normalize_D(*args):
    result = []
    for i in args:
        n = np.sum(i)
        print "initial normalization: " + str(n)
        result.append(i/n)
        assert result[-1].sum()-1 < 1e-8
    return result

def mcPurify(*args):
    result = []
    for i in args:
        assert isinstance(i,np.ndarray)
        result.append(3*np.power(i,2) - 2*np.power(i,3))
    return result
