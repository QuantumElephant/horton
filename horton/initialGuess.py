from horton.newton_rewrite import NewtonKrylov
from horton.gbasis.cext import GOBasis
from horton import System, guess_hamiltonian_core, Hamiltonian, converge_scf_oda, HartreeFock, Hartree, BeckeMolGrid, LibXCLDATerm, context, HamiltonianTerm, DenseOneBody
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

def promol_orbitals(sys, ham, basis, numChargeMult=None, ifCheat = False, charge_target=0, mult_target=None):
    orb_alpha = []
    orb_beta = []
    occ_alpha = []
    occ_beta = []
    e_alpha = []
    e_beta = []
    nbasis = []
        
    if ifCheat:
        print "CHEAT MODE ENABLED"
        fock_alpha, fock_beta = calc_DM(sys, ham, charge_target, mult_target)
        
        dm_a = sys.wfn.dm_alpha.copy()
        dm_b = sys.wfn.dm_beta.copy()
        
        orb_alpha = sys.wfn.exp_alpha
        orb_beta = sys.wfn.exp_beta
        
        occ_alpha = sys.wfn.exp_alpha.occupations
        occ_beta = sys.wfn.exp_beta.occupations
        
        e_alpha = sys.wfn.exp_alpha.energies
        e_beta = sys.wfn.exp_beta.energies
    else:
        for atomNum,i in enumerate(sys.numbers):
            atom_sys = System(np.zeros((1,3), float), np.array([i]), obasis=basis, lf = sys.lf) #hacky
            
            if isinstance(numChargeMult, np.ndarray) and atomNum in numChargeMult[:,0]:
                [num, charge, mult] = numChargeMult[np.where(numChargeMult[:,0] == atomNum),:].squeeze()
                print "REWRITING atom number " + str(num) + " element: " + str(i) + " with charge: " \
                    + str(charge) + " and multiplicity: " + str(mult)
                
                atom_sys.init_wfn(charge=charge, mult=mult,restricted=False)
            else:
                print "Calculating atomic density for element " + str(i)
                atom_sys.init_wfn(restricted=False)     
    
            nbasis.append(atom_sys.wfn.nbasis)
            guess_hamiltonian_core(atom_sys)
    
            if ham.grid is not None:
                grid = BeckeMolGrid(atom_sys, random_rotate=False) #using grid
            else:
                grid = None
            atom_ham = Hamiltonian(atom_sys, ham.terms, grid)
    
            converge_scf_oda(atom_ham, max_iter=5000)
            reset_ham(sys, ham)
            
            pro_orb_alpha.append(atom_sys.wfn.exp_alpha)
            pro_orb_beta.append(atom_sys.wfn.exp_beta)
            
            pro_occ_alpha.append(atom_sys.wfn.exp_alpha.occupations)
            pro_occ_beta.append(atom_sys.wfn.exp_beta.occupations)
            
            pro_e_alpha.append(atom_sys.wfn.exp_alpha.energies)
            pro_e_beta.append(atom_sys.wfn.exp_beta.energies)
        
        orb_alpha = sys.lf.create_one_body()
        orb_beta = sys.lf.create_one_body()
            
        orb_alpha.assign_from_blocks(*pro_orb_alpha)
        orb_beta.assign_from_blocks(*pro_orb_beta)
        
        occ_alpha = np.hstack(pro_occ_alpha)
        occ_beta = np.hstack(pro_occ_beta)
    
        e_alpha = np.hstack(pro_e_alpha)
        e_beta = np.hstack(pro_e_beta)
    
        dm_a = None
        dm_b = None
        
        fock_alpha = None
        fock_beta = None
    
    return dm_a, dm_b, fock_alpha, fock_beta, (orb_alpha, occ_alpha, e_alpha, orb_beta, occ_beta, e_beta)
    
def promol_guess(sys, orb_a, occ_a, energy_a, orb_b, occ_b, energy_b, ifFracTarget=False):
    assert occ_a.size == orb_a.nbasis and occ_b.size == orb_b.nbasis, "initial occupation size: " + str(occ_a.size) + "," + str(occ_b.size) + " basis size: " + str(orb_a.shape[0]) + "," + str(orb_b.shape[0])
    
    if ifFracTarget:
        mu_a = frac_promol_mu(energy_a, occ_a)
        mu_b = frac_promol_mu(energy_b, occ_b)
    else:
        mu_a = int_promol_mu(energy_a, occ_a)
        mu_b = int_promol_mu(energy_b, occ_b)

    N = np.sum(occ_a)
    N2 = np.sum(occ_b)
    
    pro_da, pro_ba = promol_dm_b(sys, orb_a, occ_a, energy_a, mu_a)
    pro_db, pro_bb = promol_dm_b(sys, orb_b, occ_b, energy_b, mu_b)

    return pro_da, pro_ba, pro_db, pro_bb, mu_a, mu_b, N, N2

def frac_promol_mu(energy, occ):
    raise NotImplementedError

def int_promol_mu(energy, occ):
    if energy[occ > 0.95].any():
        homo = np.array(np.max(energy[occ > 0.95]))
    else:
        print "WARNING: No occupied orbitals"
        homo = np.zeros(1)
    
    if energy[occ < 0.05].any():
        lumo = np.array(np.min(energy[occ < 0.05]))
    else:
        print "WARNING: No unoccupied orbitals"
        lumo = np.zeros(1)
    mu = (homo + lumo)/2.
    
    return mu
    
def promol_dm_b(sys, orb, occ, energy, mu):
    promol_dm = sys.lf.create_one_body()
    promol_b = sys.lf.create_one_body()
    
    promol_dm.outer(orb, occ)
    
    coeff = []
    for eval, eng in zip(occ, energy):
        if eval>0.95 or eval < 0.05:
            coeff.append((eng-mu)/(1-2*eval))
        #else: noop
    coeff = np.hstack(coeff)
    promol_b.outer(orb, coeff)
    return promol_dm, promol_b
    
def calc_DM(sys, ham, charge_target, mult_target): #TODO: remove charge + mult
    guess_hamiltonian_core(sys)
    converged = converge_scf_oda(ham, max_iter=5000)
     
    fock_alpha = sys.lf.create_one_body(sys.wfn.nbasis)
    fock_beta = sys.lf.create_one_body(sys.wfn.nbasis)
    ham.compute_fock(fock_alpha, fock_beta)
    
    return fock_alpha, fock_beta

def reset_ham(sys,ham):
    for i in ham.terms:
        if isinstance(i, HamiltonianTerm):
            i.prepare_system(sys, ham.cache, ham.grid)

def promol_frac(dm, sys):
    s = sys.get_overlap() #TODO: abstract out inverse operation in matrix
    
    p = sys.lf.create_one_body_eye()
    p.imuls(dm,s)
    
    dsds = sys.lf.create_one_body_eye()
    dsds.imuls(dm, s, dm, s)
    
    p.isubs(dsds)
    
    sinv = s.copy()
    sinv.invert()
    p.imul(sinv)
    
#     p_old = np.dot(np.dot(dm._array,s._array) - reduce(np.dot, (dm._array,s._array,dm._array,s._array)), np.linalg.inv(s._array))
#     assert np.linalg.norm(p_old - p._array) < 1e-10

    return p
    
    
def prep_D(lg, *args):
    if not lg.isTriu:
        result = [i.ravel() for i in args]
    else:
        result = []
        for i in args:
            if not isinstance(i, DenseOneBody): #should be ndarray then
                result.append(i.squeeze())
            else:
                i.iscale_diag(0.5)
                result.append(2*i.ravel())
    return np.hstack(result)

def generic_HF_calc(basis = 'sto-3g', filename='test/water_equilim.xyz'):
    system = System.from_file(context.get_fn(filename), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    ham = Hamiltonian(system, [HartreeFock()])
    return system, ham, basis
    
def generic_DFT_calc(basis = 'sto-3g', filename='test/water_equilim.xyz', lda_term = 'x'):
    system = System.from_file(context.get_fn(filename), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    grid = BeckeMolGrid(system, random_rotate=False)
    libxc_term = LibXCLDATerm(lda_term) #vwn_5
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
    return system, ham, basis

def project(orig_sys, proj_sys, *args):
    new_basis = GOBasis.concatenate(orig_sys.obasis, proj_sys.obasis)
    mixed_sys = System(orig_sys.coordinates, orig_sys.numbers, obasis = new_basis)
    mixed_sys.init_wfn(charge=0, mult=1,restricted=False)
    mixed_S = mixed_sys.get_overlap()._array
    orig_S = orig_sys.get_overlap()._array
    
    rect_S = mixed_S[:orig_S.shape[0], orig_S.shape[1]:]
    proj_inv_S = np.linalg.inv(mixed_S[orig_S.shape[0]:, orig_S.shape[1]:])
     
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
