from horton import *
import numpy as np
import matplotlib.pyplot as plt
from horton.test import common

np.set_printoptions(threshold = 2000)

def test_check_grad():
    system, ham, basis = initialGuess.generic_DFT_calc()
    S = system.get_overlap()._array

    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)

    args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")
    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)

    x0 = initialGuess.prep_D(lg, *args)

    dxs = [np.random.uniform(-0.5, 0.5, len(x0)) for i in np.arange(100)]

    common.check_delta(lg.lagrange_wrap, lg.grad_wrap, x0, dxs)

def Horton_H2O():
    basis = 'STO-3G'
#    basis = '3-21G'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    guess_hamiltonian_core(system)
#  DFT
    grid = BeckeMolGrid(system, random_rotate=False)
    libxc_term = LibXCLDATerm('c_vwn')
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

#  HF
#    ham = Hamiltonian(system, [HartreeFock()])

    converged = converge_scf_oda(ham, max_iter=5000)

# def test_HF_STO3G_H2_4():
#     solver = NewtonKrylov()
# #
#     basis = 'sto-3g'
#     system = System.from_file(context.get_fn('test/H2_4.xyz'), obasis=basis)
#     system.init_wfn(charge=0, mult=1, restricted=False)
# 
#     dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
#     occ_a = np.array([0.5,0.5]); N=1 #STO-3G ONLY
#     occ_b = np.array([0.5,0.5]); N2=1 #STO-3G ONLY
#     pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
# 
# #HF
#     ham = Hamiltonian(system, [HartreeFock()])
# 
#     args = [pro_da, pro_ba, pro_db, pro_bb, mua, mub]
# 
#     norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
#     norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")
# 
#     lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = False)
# 
#     x0 = initialGuess.prep_D(lg, *args)
# 
#     print "start HF_STO3G"
#     x_star = solver.solve(lg, x0)
# 
#     print "Actual E:" + str(-74.965901) #NIST
#     print "Computed E:" + str(ham.compute_energy())
# #    assert np.abs(ham.compute_energy() - -74.965901) < 1e-4


# def test_DFT_STO3G_H2_4(): #KNOWN TO FAIL CONVERGENCE
#     solver = NewtonKrylov()
# #
#     basis = 'sto-3g'
#     system = System.from_file(context.get_fn('test/H2_4.xyz'), obasis=basis)
#     system.init_wfn(charge=0, mult=1, restricted=False)
#
#     dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
#     occ_a = np.array([0.5,0.5]); N=1 #STO-3G ONLY
#     occ_b = np.array([0.5,0.5]); N2=1 #STO-3G ONLY
#     pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
#
#     grid = BeckeMolGrid(system, random_rotate=False)
#
#     libxc_term = LibXCLDATerm('c_vwn') #vwn_5
#     ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
#
#     args = [pro_da, pro_ba, pro_db, pro_bb, mua, mub]
#
#     norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
#     norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")
#
#     lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = False)
#
#     x0 = initialGuess.prep_D(lg, *args)
#
#     print "start HF_STO3G"
#     x_star = solver.solve(lg, x0)
#
#     print "Actual E:" + str(-74.965901) #NIST
#     print "Computed E:" + str(ham.compute_energy()) #KNOWN TO FAIL CONVERGENCE
# #    assert np.abs(ham.compute_energy() - -74.965901) < 1e-4


# def test_DFT_STO3G_Frac_H2_4():
#     solver = NewtonKrylov()
# #
#     basis = 'sto-3g'
#     system = System.from_file(context.get_fn('test/H2_4.xyz'), obasis=basis)
#     system.init_wfn(charge=0, mult=1, restricted=False)
# 
#     dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis, np.array([[1,-1,1]]))
#     occ_a = np.array([0.5,0.5]); N=1 #STO-3G ONLY
#     occ_b = np.array([0.5,0.5]); N2=1 #STO-3G ONLY
#     pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
#     pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)
# 
#     grid = BeckeMolGrid(system, random_rotate=False)
# 
#     libxc_term = LibXCLDATerm('c_vwn') #vwn_5
#     ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
# 
#     args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]
# 
#     L = system.lf.create_one_body_from(np.eye(dm_a.shape[0]))
# 
#     norm_a = LinearConstraint(system, N, L, select="alpha")
#     norm_b = LinearConstraint(system, N2, L, select="beta")
# 
#     lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
# 
#     x0 = initialGuess.prep_D(lg, *args)
# 
#     print "Start DFT_STO3G_Frac"
#     x_star = solver.solve(lg, x0)
# 
#     print "Actual E:" + str(-66.634688718437) #NWCHEM
#     print "Computed E:" + str(ham.compute_energy())
# #    assert np.abs(ham.compute_energy() - -66.634688718437) < 1e-4

def test_linear_stepped_constraints():
    solver = NewtonKrylov()
#
    basis = 'sto-3g'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)

    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis, ifCheat = True)
    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)

    grid = BeckeMolGrid(system, random_rotate=False)

    libxc_term = LibXCLDATerm('c_vwn') #vwn_5
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
#     ham = Hamiltonian(system, [HartreeFock()])

    args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]

    norm_a = LinearConstraint(system, N+0.5, np.eye(dm_a.shape[0]), select="alpha", C_init=N, steps=50)
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

#     norm_b = LinearConstraint(system, N2-0.5, np.eye(dm_a.shape[0]), select="beta", C_init=N, steps=50)

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)

    x0 = initialGuess.prep_D(lg, *args)

    print "Start DFT_STO3G_Frac"
    x_star = solver.fancy_solve(lg, x0)

    print "Actual E:" + str(73.6549343469) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
#     assert np.abs(ham.compute_energy() - -73.6549343469) < 1e-4

def test_quadratic_stepped_constraints():
    solver = NewtonKrylov()
    sys, ham, basis = initialGuess.generic_DFT_calc(lda_term = "c_vwn")

    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(sys, basis, ifCheat = True)
    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(sys, pro_da, pro_db)

    args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]

    norm_a = QuadraticConstraint(sys, N+0.5, [np.eye(dm_a.shape[0]), np.eye(dm_a.shape[0])], select="alpha", D=pro_da, steps=50)
    norm_b = QuadraticConstraint(sys, N2-0.5, [np.eye(dm_a.shape[0]), np.eye(dm_a.shape[0])], select="beta", D=pro_db, steps=50)

    lg = Lagrangian(sys, ham, [norm_a, norm_b], isFrac = True)

    x0 = initialGuess.prep_D(lg, *args)

    print "Start DFT_STO3G_Frac"
    x_star = solver.fancy_solve(lg, x0)

    print "Actual E:" + str(73.6549343469) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())

def projected_h2o_calc(origBasis, projBasis, method, targetE, ifCheat=False, isFrac=False):
    if method == "DFT":
        Exc = "c_vwn"
    else:
        Exc = None
    origSys, origHam, origArgs, origOpt = setup_system(origBasis, method, file='test/water_equilim.xyz',
                                                       Exc=Exc, ifCheat=True, isFrac=isFrac)
    projSys, projHam, cheatArgs, projOpt = setup_system(projBasis, method, file='test/water_equilim.xyz',
                                                        Exc=Exc, ifCheat=True, isFrac=isFrac)
    projArgs = initialGuess.project(origSys, projSys, *origArgs)
    
#     print [np.linalg.norm(i-j) for i,j in zip(cheatArgs, projArgs)]
    
    #regenerating projection using updated fock matrix
    projD_alpha = projArgs[0]
    if isFrac:
        projD_beta = projArgs[3]
    else:
        projD_beta = projArgs[2]

    setup_wfn(projSys, projHam, projD_alpha, projD_beta)
    
    projArgs, N, N2 = setup_guess(projSys, (projSys.wfn.exp_alpha._coeffs, projSys.wfn.exp_alpha.occupations,
                                            projSys.wfn.exp_alpha.energies, projSys.wfn.exp_beta._coeffs, 
                                            projSys.wfn.exp_beta.occupations, projSys.wfn.exp_beta.energies),
                                  isFrac=isFrac)
    
    print [np.linalg.norm(i-j) for i,j in zip(cheatArgs, projArgs)]
    
    cons = setup_cons(projSys, projArgs, N, N2)
    x_star = setup_lg(projSys, projHam, cons, projArgs, projBasis, method, isFrac=isFrac)
    check_E(projHam, targetE)

def setup_wfn(sys, ham, dm_alpha, dm_beta):    
    sys.wfn.update_dm("alpha", sys.lf.create_one_body_from(dm_alpha))
    sys.wfn.update_dm("beta", sys.lf.create_one_body_from(dm_beta))
    
    fock_alpha = sys.lf.create_one_body(sys.wfn.nbasis)
    fock_beta = sys.lf.create_one_body(sys.wfn.nbasis)
    ham.compute_fock(fock_alpha, fock_beta)
    projSys.wfn.invalidate()
    projSys.wfn.update_exp(fock_alpha, fock_beta, projSys.get_overlap(), projD_alpha, projD_beta) #incase we have fractional occ

def default_h2o_calc(basis, method, targetE, ifCheat=False, isFrac=False):
    if method == "DFT":
        Exc = "c_vwn"
    else:
        Exc = None
    sys, ham, args, opt = setup_system(basis, method, file='test/water_equilim.xyz', 
                                           Exc=Exc, ifCheat=ifCheat, isFrac=isFrac)
    cons = setup_cons(sys, args, N=opt["N"], N2=opt["N2"])
    x_star = setup_lg(sys, ham, cons, args, basis, method, isFrac=isFrac)
    check_E(ham, targetE)
    
# def frac_target_h2o_calc(basis, method, targetE, ifCheat=False, isFrac=False):
#     if method == "DFT":
#         Exc = "c_vwn"
#     else:
#         Exc = None
#     low_sys, low_ham, low_args, low_opt = setup_system(basis, method, file='test/water_equilim.xyz', 
#                                            Exc=Exc, ifCheat=True, isFrac=isFrac, 
#                                            Ntarget_alpha = 4, Ntarget_beta=4)
#     high_sys, high_ham, high_args, high_opt = setup_system(basis, method, file='test/water_equilim.xyz', 
#                                            Exc=Exc, ifCheat=True, isFrac=isFrac, 
#                                            Ntarget_alpha = 5, Ntarget_beta=5)
#     
#     frac_sys, frac_ham, frac_args, frac_opt = setup_system(basis, method, file='test/water_equilim.xyz', 
#                                            Exc=Exc, ifCheat=True, isFrac=isFrac, 
#                                            Ntarget_alpha = 4.5, Ntarget_beta=4.5)
#     
#     fracDM_alpha = (low_args[0]+high_args[0])/2.
#     if isFrac:
#         fracDM_beta = (low_args[3]+high_args[3])/2.
#     else:
#         fracDM_beta = (low_args[2]+high_args[2])/2.  
#      
#     setup_wfn(frac_sys, frac_ham, fracDM_alpha, fracDM_beta)
#     frac_args, N, N2 = setup_guess(frac_sys, (frac_sys.wfn.exp_alpha._coeffs, frac_sys.wfn.exp_alpha.occupations,
#                                             frac_sys.wfn.exp_alpha.energies, frac_sys.wfn.exp_beta._coeffs, 
#                                             frac_sys.wfn.exp_beta.occupations, frac_sys.wfn.exp_beta.energies),
#                                   isFrac=isFrac)
#     
#     cons = setup_cons(frac_sys, frac_args, N=frac_opt["N"], N2=frac_opt["N2"])
#     x_star = setup_lg(frac_sys, frac_ham, cons, frac_args, basis, method, isFrac=isFrac)
#     check_E(ham, targetE)

def frac_target_h2o_calc(basis, method, targetE, ifCheat=False, isFrac=False):
    if method == "DFT":
        Exc = "c_vwn"
    else:
        Exc = None
    sys, ham, args, opt = setup_system(basis, method, file='test/water_equilim.xyz', 
                                           Exc=Exc, ifCheat=ifCheat, isFrac=isFrac)
    
    print sys.wfn.exp_alpha.occ
    
    cons = setup_cons(sys, args, N=opt["N"], N2=opt["N2"])
    x_star = setup_lg(sys, ham, cons, args, basis, method, isFrac=isFrac)
    check_E(ham, targetE)

def setup_system(basis, method, file, ifCheat = False, isFrac = False, 
                     restricted=False, Exc = None, random_rotate=False, 
                     Ntarget_alpha=None, Ntarget_beta=None):
    lf = matrix.TriangularLinalgFactory()
#     lf = matrix.DenseLinalgFactory()
    system = System.from_file(context.get_fn(file), obasis=basis, lf=lf)
    if Ntarget_alpha is not None and Ntarget_beta is not None:
        
        print "overriding alpha population: " + str(Ntarget_alpha)
        print 'overriding beta population: ' + str(Ntarget_beta)
        
        N = Ntarget_alpha
        N2 = Ntarget_beta
        
        Ntarget=Ntarget_alpha + Ntarget_beta
        charge_target = system.numbers.sum() - Ntarget
    elif Ntarget_alpha is not None or Ntarget_beta is not None:
        assert False, "must define both alpha and beta target occupations."
    else:
        charge_target = 0
    
    system.init_wfn(charge=charge_target, restricted=restricted)
    if method == "HF":
        ham = Hamiltonian(system, [HartreeFock()])
        assert Exc is None
    elif method == "DFT":
        assert Exc is not None
        if not random_rotate:
            print "Random grid rotation disabled."
        grid = BeckeMolGrid(system, random_rotate=random_rotate)
        libxc_term = LibXCLDATerm(Exc)
        ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
    else:
        assert False, "not a valid level of theory"
    dm_a, dm_b, fock_alpha, fock_beta, promol_args = initialGuess.promol_orbitals(system, ham, basis, 
                                                                                  ifCheat=ifCheat, charge_target=charge_target)

    args, N, N2 = setup_guess(system, promol_args, dm_a, dm_b, isFrac=isFrac)
    
    if Ntarget_alpha is not None and Ntarget_beta is not None:
        assert np.abs(N-Ntarget_alpha) < 1e-13, (Ntarget_alpha, N)
        assert np.abs(N2-Ntarget_beta) < 1e-13, (Ntarget_beta, N2)

    return system, ham, args, locals()

def setup_guess(system, promol_args, dm_a=None, dm_b=None, isFrac=False):
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(*promol_args)
    
    if dm_a is not None:
        pro_da = dm_a #CHEATING 
    if dm_b is not None:
        pro_db = dm_b #CHEATING
        
    if isFrac:
        pa = initialGuess.promol_frac(pro_da, system)
        pb = initialGuess.promol_frac(pro_db, system)

        args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]
    else:
        args = [pro_da, pro_ba, pro_db, pro_bb, mua, mub]
        
    return args, N, N2
    
def setup_lg(sys, ham, cons, args, basis, method, Exc=None, isFrac=False):
    if isFrac:
        assert len(args) > 6
    
    solver = NewtonKrylov()
    lg = Lagrangian(sys, ham, cons, isFrac=isFrac)
    
    x0 = initialGuess.prep_D(lg, *args)

    msg = "Start " + method +" "
    if Exc is not None:
        msg += Exc + " "
    msg += basis + " "
    if isFrac:
        msg += "Fractional "
    else:
        msg += "Integer "
    print msg
    
    x_star = solver.solve(lg, x0)
    
    return x_star
    
def setup_cons(sys, args, N, N2):
    L = sys.lf.create_one_body_from(np.eye(sys.wfn.nbasis))

    norm_a = LinearConstraint(sys, N, L, select="alpha")
    norm_b = LinearConstraint(sys, N2, L, select="beta")

    return [norm_a, norm_b]

def check_E(ham, targetE):
    print "Actual E:" + str(targetE) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - targetE) < 1e-4

# test_check_grad()
# test_linear_stepped_constraints()
# test_quadratic_stepped_constraints()
# test_HF_STO3G_H2_4()
# # test_DFT_STO3G_H2_4()
# test_DFT_STO3G_Frac_H2_4()

# default_h2o_calc('sto-3g', "HF", -74.965901, ifCheat=True) #NWCHEM
# default_h2o_calc('3-21G', "HF", -75.583747447860, ifCheat=True) #NWCHEM
# default_h2o_calc('6-31++G**', "HF", -76.0483902449622, ifCheat=True) #Horton
# default_h2o_calc('sto-3g', "DFT", -66.634688718437, ifCheat=True) #NWCHEM
# default_h2o_calc('3-21G', "DFT", -67.521923845983, ifCheat=True) #NWCHEM

# default_h2o_calc('sto-3g', "HF", -74.965901, ifCheat=True, isFrac=True) #NWCHEM
# default_h2o_calc('3-21G', "HF", -75.583747447860, ifCheat=True, isFrac=True) #NWCHEM
default_h2o_calc('sto-3g', "DFT", -66.634688718437, ifCheat=True, isFrac=True) #NWCHEM
# default_h2o_calc('3-21G', "DFT", -67.521923845983, ifCheat=True, isFrac=True) #NWCHEM

# projected_h2o_calc('sto-3g', '6-31G', "DFT", -67.9894175486548, ifCheat=True, isFrac=True) #Horton

# frac_target_h2o_calc('sto-3g', "DFT", -66.634688718437, ifCheat=True, isFrac=True) #NWCHEM
