from horton.gbasis.cext import GOBasis
from horton.meanfield import guess_hamiltonian_core, Observable, converge_scf_oda, builtin, LibXCLDA, wfn, Hamiltonian, HartreeFockExchange, Hartree, UnrestrictedWFN
from horton import System, BeckeMolGrid, DenseOneBody, matrix, context, log, LinearConstraint, Lagrangian
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import copy as cp

np.set_printoptions(threshold = 2000)

def setup_system(basis, method, file, ifCheat = False, isFrac = False, 
                     restricted=False, Exc = None, random_rotate=False, 
                     Ntarget_alpha=None, Ntarget_beta=None):
#     lf = matrix.IVDualLinalgFactory()
#     lf = matrix.TriangularLinalgFactory()
    lf = matrix.DenseLinalgFactory()
#     lf = matrix.MPDualLinalgFactory()
    
    system = System.from_file(context.get_fn(file), obasis=basis, lf=lf)
#     ####TESTING
#     try:
#         system.lf.enable_dual()
#     except AttributeError:
#         pass
#     #TESTING####
    if Ntarget_alpha is not None and Ntarget_beta is not None:
        if log.do_medium:
            log("overriding alpha population: " + str(Ntarget_alpha))
            log('overriding beta population: ' + str(Ntarget_beta))
        
        N = Ntarget_alpha
        N2 = Ntarget_beta
        
        Ntarget=Ntarget_alpha + Ntarget_beta
        charge_target = system.numbers.sum() - Ntarget
    elif Ntarget_alpha is not None or Ntarget_beta is not None:
        assert False, "must define both alpha and beta target occupations."
    else:
        charge_target = 0
    
    wfn.setup_mean_field_wfn(system, charge=charge_target, restricted=restricted)
    if method == "HF":
        ham = Hamiltonian(system, [HartreeFockExchange()])
        assert Exc is None
    elif method == "DFT":
        assert Exc is not None
        if not random_rotate:
            log("Random grid rotation disabled.")
        grid = BeckeMolGrid(system, random_rotate=random_rotate)
        libxc_term = LibXCLDA(Exc)
        ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
    else:
        assert False, "not a valid level of theory"
    promol_orbitals(system, ham, basis, ifCheat=ifCheat, charge_target=charge_target)
    args, N, N2 = setup_guess(system, ifCheat=True, isFrac=isFrac)
    
    if Ntarget_alpha is not None and Ntarget_beta is not None:
        assert np.abs(N-Ntarget_alpha) < 1e-13, (Ntarget_alpha, N)
        assert np.abs(N2-Ntarget_beta) < 1e-13, (Ntarget_beta, N2)

    return system, ham, args, locals()

def setup_guess(system, ifCheat=False, isFrac=False):
    args, N, N2 = promol_guess(system)

    if ifCheat:
        pro_da = system.wfn.dm_alpha
        if isinstance(system.wfn, UnrestrictedWFN):
            pro_db = system.wfn.dm_beta
        else:
            pro_db = system.wfn.dm_alpha.copy()
        
    if isFrac:
        pa = promol_frac(pro_da, system)
        pb = promol_frac(pro_db, system)

        args.insert(2,pa)
        args.insert(5,pb)
        
    return args, N, N2
    
def setup_lg(sys, ham, cons, args, basis, method, Exc=None, isFrac=False):
    sys.wfn.clear() #Safety check. Make sure we aren't computing with cheat values.
    ham.clear()
    
    if isFrac:
        assert len(args) > 6

    lg = Lagrangian(sys, ham, cons, isFrac=isFrac, ifHess=True)

    x0 = lg.matHelper.initialize(*args)

    msg = "Start " + method +" "
    if Exc is not None:
        msg += Exc + " "
    msg += basis + " "
    if isFrac:
        msg += "Fractional "
    else:
        msg += "Integer "
    if log.do_medium:
        log(msg)
    return lg, x0
    
def setup_cons(sys, args, N, N2):
    L = sys.lf.create_one_body_from(np.eye(sys.wfn.nbasis))

    norm_a = LinearConstraint(sys, N, L, select="alpha")
    norm_b = LinearConstraint(sys, N2, L, select="beta")

    return [norm_a, norm_b]

def check_E(ham, targetE):
    if log.do_medium:
        log()
        ham.log_energy()
        log()
        log("Actual E:" + str(targetE)) #NWCHEM
        log("Computed E:" + str(ham.compute()))
    assert np.abs(ham.compute() - targetE) < 1e-4

def sqrtm(A):
    result = np.zeros_like(A)
    eigvs,eigvc = np.linalg.eigh(A)
    for i in np.arange(len(eigvs)):
        if eigvs[i] > 0:
            result += np.sqrt(eigvs[i])*np.outer(eigvc[i], eigvc[i])
    
    return result

def promol_orbitals(sys, ham, basis, numChargeMult=None, ifCheat = False, charge_target=0, mult_target=None):
    if ifCheat:
        print "CHEAT MODE ENABLED"
        calc_DM(sys, ham, charge_target, mult_target)
        
    else:
        pro_orb_alpha = []
        pro_occ_alpha = []
        pro_e_alpha = []
        if isinstance(sys.wfn, UnrestrictedWFN):
            pro_orb_beta = []
            pro_occ_beta = []
            pro_e_beta = []
            
        for atomNum,i in enumerate(sys.numbers):
            atom_sys = System(np.zeros((1,3), float), np.array([i]), obasis=basis) #hacky
            
            if isinstance(numChargeMult, np.ndarray) and atomNum in numChargeMult[:,0]:
                [num, charge, mult] = numChargeMult[np.where(numChargeMult[:,0] == atomNum),:].squeeze()
                print "REWRITING atom number " + str(num) + " element: " + str(i) + " with charge: " \
                    + str(charge) + " and multiplicity: " + str(mult)
                
                wfn.setup_mean_field_wfn(atom_sys, charge=charge_target, mult=mult, restricted=False)
            else:
                print "Calculating atomic density for element " + str(i)
                wfn.setup_mean_field_wfn(atom_sys, restricted=False)     
    
            nbasis.append(atom_sys.wfn.nbasis)
            guess_hamiltonian_core(atom_sys)
    
            if ham.grid is not None:
                grid = BeckeMolGrid(atom_sys, random_rotate=False) #using grid
            else:
                grid = None
            atom_ham = Hamiltonian(atom_sys, [HartreeFockExchange()]) #TODO: generalize for DFT or HF
    
            converge_scf_oda(atom_ham, maxiter=5000)
            
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
        
        sys.wfn.exp_alpha._coeffs = orb_alpha._to_numpy()
        sys.wfn.exp_beta._coeffs = orb_beta._to_numpy()
        
        sys.wfn.exp_alpha.occupations = np.hstack(pro_occ_alpha)
        sys.wfn.exp_beta.occupations = np.hstack(pro_occ_beta)
    
        sys.wfn.exp_alpha.energies = np.hstack(pro_e_alpha)
        sys.wfn.exp_beta.energies = np.hstack(pro_e_beta)
    
def promol_guess(sys, ifFracTarget=False):

    if isinstance(sys.wfn, UnrestrictedWFN):
        beta_exp = sys.wfn.exp_beta
    else:
        beta_exp = sys.wfn.exp_alpha 
    
    if ifFracTarget:
        mu_a = frac_promol_mu(sys.wfn.exp_alpha)
        mu_b = frac_promol_mu(beta_exp)
    else:
        mu_a = int_promol_mu(sys.wfn.exp_alpha)
        mu_b = int_promol_mu(beta_exp)

    N = np.sum(sys.wfn.exp_alpha.occupations)
    N2 = np.sum(beta_exp.occupations)
    
    pro_da, pro_ba = promol_dm_b(sys, sys.wfn.exp_alpha, mu_a)
    pro_db, pro_bb = promol_dm_b(sys, beta_exp, mu_b)

    return [pro_da, pro_ba, pro_db, pro_bb, mu_a, mu_b], N, N2

def frac_promol_mu(exp):
    raise NotImplementedError

def int_promol_mu(exp):
    if exp.energies[exp.occupations > 0.95].any():
        homo = np.array(np.max(exp.energies[exp.occupations > 0.95]))
    else:
        print "WARNING: No occupied orbitals"
        homo = np.zeros(1)
    
    if exp.energies[exp.occupations < 0.05].any():
        lumo = np.array(np.min(exp.energies[exp.occupations < 0.05]))
    else:
        print "WARNING: No unoccupied orbitals"
        lumo = np.zeros(1)
    mu = (homo + lumo)/2.
    
    return mu
    
def promol_dm_b(sys, exp, mu):
    promol_dm = sys.lf.create_one_body()
    promol_b = sys.lf.create_one_body()
    
    promol_dm.outer(exp, exp.occupations)
    
    coeff = []
    for eval, eng in zip(exp.occupations, exp.energies):
        if eval>0.95 or eval < 0.05:
            coeff.append((eng-mu)/(1-2*eval))
        #else: noop
    coeff = np.hstack(coeff)
    promol_b.outer(exp, coeff)
    return promol_dm, promol_b
    
def calc_DM(sys, ham, charge_target, mult_target): #TODO: remove charge + mult
    guess_hamiltonian_core(sys)
    converged = converge_scf_oda(ham, maxiter=5000)
     
    fock_alpha = sys.lf.create_one_body(sys.wfn.nbasis)
    fock_beta = sys.lf.create_one_body(sys.wfn.nbasis)
    ham.compute_fock(fock_alpha, fock_beta)
    
    return fock_alpha, fock_beta

def reset_ham(sys,ham):
    for i in ham.terms:
        if isinstance(i, Observable):
            i.prepare_system(sys, ham.cache, ham.grid)

def promol_frac(dm, sys):
    s = sys.get_overlap() #TODO: abstract out inverse operation in matrix
    
    p = sys.lf.create_one_body_eye()
    p.imuls(s,dm,s)
    
    dsds = sys.lf.create_one_body_eye()
    dsds.imuls(s,dm, s, dm, s)
    
    p.isubs(dsds)
    
    sinv = s.copy()
    sinv.invert()
    p.imul(sinv)
    
#     p_old = np.dot(np.dot(dm._array,s._array) - reduce(np.dot, (dm._array,s._array,dm._array,s._array)), np.linalg.inv(s._array))
#     assert np.linalg.norm(p_old - p._array) < 1e-10

    return p

def generic_HF_calc(basis = 'sto-3g', filename='test/water_equilim.xyz'):
    system = System.from_file(context.get_fn(filename), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    ham = Hamiltonian(system, [HartreeFock()])
    return system, ham, basis
    
def generic_DFT_calc(basis = 'sto-3g', filename='test/water_equilim.xyz', lda_term = 'x'):
    system = System.from_file(context.get_fn(filename), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    grid = BeckeMolGrid(system, random_rotate=False)
    libxc_term = LibXCLDA(lda_term) #x
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
