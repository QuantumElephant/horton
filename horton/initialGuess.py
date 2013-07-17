from horton.newton_rewrite import NewtonKrylov
from horton.gbasis.cext import GOBasis
from horton import System, guess_hamiltonian_core, Hamiltonian, converge_scf_oda, HartreeFock, Hartree, BeckeMolGrid, LibXCLDATerm, context
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

def promol_orbitals(sys, ham, basis, numChargeMult=None, ifCheat = False):
    orb_alpha = []
    orb_beta = []
    occ_alpha = []
    occ_beta = []
    e_alpha = []
    e_beta = []
    nbasis = []
        
    if ifCheat:
        print "CHEAT MODE ENABLED"
        cheatSys = calc_DM(sys, ham)
        
        dm_a = cheatSys.wfn.dm_alpha._array
        dm_b = cheatSys.wfn.dm_beta._array
        
        orb_alpha = cheatSys.wfn.exp_alpha._coeffs
        orb_beta = cheatSys.wfn.exp_beta._coeffs
        
        occ_alpha = cheatSys.wfn.exp_alpha.occupations
        occ_beta = cheatSys.wfn.exp_beta.occupations
        
        e_alpha = cheatSys.wfn.exp_alpha.energies
        e_beta = cheatSys.wfn.exp_beta.energies
    else:
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
    
            if ham.grid is not None:
                grid = BeckeMolGrid(atom_sys, random_rotate=False)
            else:
                grid = None
            atom_ham = Hamiltonian(atom_sys, ham.terms, grid)
    
            converge_scf_oda(atom_ham, max_iter=5000)
            
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
    
        dm_a = None
        dm_b = None
    
    return dm_a, dm_b, orb_alpha, orb_beta, occ_alpha, occ_beta, e_alpha, e_beta, nbasis
    
def promol_guess(orb_a, orb_b, occ_a, occ_b, energy_a, energy_b, N = None, N2 = None):
    assert occ_a.size == orb_a.shape[0] and occ_b.size == orb_b.shape[0], "initial occupation size: " + str(occ_a.size) + "," + str(occ_b.size) + " basis size: " + str(orb_a.shape[0]) + "," + str(orb_b.shape[0])
    
    mu_a = int_promol_mu(energy_a, occ_a)
    mu_b = int_promol_mu(energy_b, occ_b)

    if N is None or N2 is None:
        N = np.sum(occ_a)
        N2 = np.sum(occ_b)
    
    pro_da, pro_ba = promol_dm_b(orb_a, occ_a, energy_a, mu_a)
    pro_db, pro_bb = promol_dm_b(orb_b, occ_b, energy_b, mu_b)

    return pro_da, pro_ba, pro_db, pro_bb, mu_a, mu_b, N, N2

def int_promol_mu(energy, occ):
    homo = np.array(np.max(energy[occ > 0.95]))
    lumo = np.array(np.min(energy[occ < 0.05]))
    mu = (homo + lumo)/2.
    
    return mu
    
def promol_dm_b(orb, occ, energy, mu):
    promol_dm = np.zeros_like(orb)
    promol_b = np.zeros_like(orb)
    
    for eval, evec, eng in zip(occ, orb.T, energy):
        outer = np.outer(evec, evec)
        promol_dm += outer*eval
        
        if eval>0.95 or eval < 0.05:
            coeff = (eng-mu)/(1-2*eval)
        else:
            coeff = 0
        promol_b += outer*coeff
        
    return promol_dm, promol_b
    
def calc_DM(sys,ham):
    system = System(sys.coordinates, sys.numbers, obasis=sys.obasis)
    system.init_wfn(restricted=False) #TODO: generalize for closed shells systems
    guess_hamiltonian_core(system)
    ham_copy = Hamiltonian(system, ham.terms, ham.grid)
        
    converged = converge_scf_oda(ham_copy, max_iter=5000)
    return system

def promol_frac_old(orb, occ):
    promol_p = np.zeros_like(orb)
    
    for eval, evec in zip(occ, orb.T):
        outer = np.outer(evec, evec)
        pre_coeff = eval*(1-eval)
        if pre_coeff > 0:
            coeff = np.sqrt(pre_coeff)
        else:
            coeff=0
        promol_p += outer*coeff 

    return promol_p

def promol_frac(dm, sys):
    s = sys.get_overlap()._array #TODO: abstract out inverse operation in matrix
    
    p = np.dot(np.dot(dm,s) - reduce(np.dot, (dm,s,dm,s)), np.linalg.inv(s))
    return p
    
    
def prep_D(lg, *args):
    if not lg.isTriu:
        result = [i.ravel() for i in args]
    else:
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

def generic_HF_calc(basis = 'sto-3g', filename='test/water_equilim.xyz'):
    system = System.from_file(context.get_fn(filename), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    ham = Hamiltonian(system, [HartreeFock()])
    return system, ham, basis
    
def generic_DFT_calc(basis = 'sto-3g', filename='test/water_equilim.xyz', lda_term = 'c_vwn'):
    system = System.from_file(context.get_fn(filename), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    grid = BeckeMolGrid(system, random_rotate=False)
    libxc_term = LibXCLDATerm(lda_term) #vwn_5
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
    return system, ham, basis

def calc_shapes(*args):
    shapes = []
    for i in args:
        if i.size == 1:
            shapes.append(i.size)
            continue
        shapes.append(i.shape[0])
    return shapes

def project(orig_sys, proj_sys, *args):
    new_basis = GOBasis.concatenate(orig_sys.obasis, proj_sys.obasis)
    mixed_sys = System(orig_sys.coordinates, orig_sys.numbers, obasis = new_basis)
    mixed_sys.init_wfn(charge=0, mult=1,restricted=False)
    mixed_S = mixed_sys.get_overlap()._array
    orig_S = orig_sys.get_overlap()._array
    
    rect_S = mixed_S[:orig_S.shape[0], orig_S.shape[1]:]
    proj_inv_S = np.linalg.pinv(mixed_S[orig_S.shape[0]:, orig_S.shape[1]:])
     
    result = []
    for i in args:
        if not isinstance(i, np.ndarray) or i.size == 1:
            print "DEBUG: skipping projection of size 1 matrix. Appending unprojected to result."
            result.append(i)
            continue
        
        result.append(reduce(np.dot,[proj_inv_S,rect_S.T,i,rect_S,proj_inv_S]))
    return result

def normalize_D(sys, D, N):
    S = sys.get_overlap()._array
    n = np.dot(S.ravel(), D.ravel())
    print "initial normalization: " + str(n)
    result = D*N/n
    return result

def mcPurify(sys, *args):
    result = []
    for i in args:
        assert isinstance(i,np.ndarray)
        result.append(3*np.power(np.dot(sys.get_overlap()._array, i),2) - 2*np.power(np.dot(sys.get_overlap()._array, i),3))
    return result
