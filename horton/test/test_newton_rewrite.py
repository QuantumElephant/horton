from horton import *
from horton.meanfield import *
import numpy as np
import matplotlib.pyplot as plt
from horton.test import common

np.set_printoptions(threshold = 2000)

# def test_check_grad():
#     system, ham, basis = initialGuess.generic_DFT_calc()
#     S = system.get_overlap()._array
# 
#     dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
#     occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
#     occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
#     pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
#     pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)
# 
#     args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]
# 
#     norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
#     norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")
#     lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
# 
#     x0 = initialGuess.prep_D(lg, *args)
# 
#     dxs = [np.random.uniform(-0.5, 0.5, len(x0)) for i in np.arange(100)]
# 
#     common.check_delta(lg.lagrange_wrap, lg.grad_wrap, x0, dxs)

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

# def test_linear_stepped_constraints():
#     solver = NewtonKrylov()
# #
#     basis = 'sto-3g'
#     system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
#     system.init_wfn(charge=0, mult=1, restricted=False)
# 
#     dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis, ifCheat = True)
#     occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
#     occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
#     pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
#     pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)
# 
#     grid = BeckeMolGrid(system, random_rotate=False)
# 
#     libxc_term = LibXCLDATerm('c_vwn') #vwn_5
#     ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
# #     ham = Hamiltonian(system, [HartreeFock()])
# 
#     args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]
# 
#     norm_a = LinearConstraint(system, N+0.5, np.eye(dm_a.shape[0]), select="alpha", C_init=N, steps=50)
#     norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")
# 
# #     norm_b = LinearConstraint(system, N2-0.5, np.eye(dm_a.shape[0]), select="beta", C_init=N, steps=50)
# 
#     lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
# 
#     x0 = initialGuess.prep_D(lg, *args)
# 
#     print "Start DFT_STO3G_Frac"
#     x_star = solver.fancy_solve(lg, x0)
# 
#     print "Actual E:" + str(73.6549343469) #NWCHEM
#     print "Computed E:" + str(ham.compute())
# #     assert np.abs(ham.compute() - -73.6549343469) < 1e-4
# 
# def test_quadratic_stepped_constraints():
#     solver = NewtonKrylov()
#     sys, ham, basis = initialGuess.generic_DFT_calc(lda_term = "x")
# 
#     dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(sys, basis, ifCheat = True)
#     occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
#     occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
#     pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
#     pa, pb = initialGuess.promol_frac(sys, pro_da, pro_db)
# 
#     args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]
# 
#     norm_a = QuadraticConstraint(sys, N+0.5, [np.eye(dm_a.shape[0]), np.eye(dm_a.shape[0])], select="alpha", D=pro_da, steps=50)
#     norm_b = QuadraticConstraint(sys, N2-0.5, [np.eye(dm_a.shape[0]), np.eye(dm_a.shape[0])], select="beta", D=pro_db, steps=50)
# 
#     lg = Lagrangian(sys, ham, [norm_a, norm_b], isFrac = True)
# 
#     x0 = initialGuess.prep_D(lg, *args)
# 
#     print "Start DFT_STO3G_Frac"
#     x_star = solver.fancy_solve(lg, x0)
# 
#     print "Actual E:" + str(73.6549343469) #NWCHEM
#     print "Computed E:" + str(ham.compute())

def projected_h2o_calc(origBasis, projBasis, method, targetE, ifCheat=False, isFrac=False):
    if method == "DFT":
        Exc = "x"
    else:
        Exc = None
    origSys, origHam, origArgs, origOpt = initialGuess.setup_system(origBasis, method, file='test/water_equilim.xyz',
                                                       Exc=Exc, ifCheat=True, isFrac=isFrac)
    projSys, projHam, cheatArgs, projOpt = initialGuess.setup_system(projBasis, method, file='test/water_equilim.xyz',
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
    
    projArgs, N, N2 = initialGuess.setup_guess(projSys, (projSys.wfn.exp_alpha._coeffs, projSys.wfn.exp_alpha.occupations,
                                            projSys.wfn.exp_alpha.energies, projSys.wfn.exp_beta._coeffs, 
                                            projSys.wfn.exp_beta.occupations, projSys.wfn.exp_beta.energies),
                                  isFrac=isFrac)
    
    print [np.linalg.norm(i-j) for i,j in zip(cheatArgs, projArgs)]
    
    cons = initialGuess.setup_cons(projSys, projArgs, N, N2)
    lg,x0 = initialGuess.setup_lg(projSys, projHam, cons, projArgs, projBasis, method, isFrac=isFrac)
    x_star, nIter = solver.solve(lg, x0)
    initialGuess.check_E(projHam, targetE)

def setup_wfn(sys, ham, dm_alpha, dm_beta):    
    sys.wfn.update_dm("alpha", sys.lf.create_one_body_from(dm_alpha))
    sys.wfn.update_dm("beta", sys.lf.create_one_body_from(dm_beta))
    
    fock_alpha = sys.lf.create_one_body(sys.wfn.nbasis)
    fock_beta = sys.lf.create_one_body(sys.wfn.nbasis)
    ham.compute_fock(fock_alpha, fock_beta)
    sys.wfn.clear()
    sys.wfn.update_exp(fock_alpha, fock_beta, ham.overlap, 
                       sys.lf.create_one_body_from(dm_alpha), 
                       sys.lf.create_one_body_from(dm_beta)) #incase we have fractional occ

def default_h2o_calc(basis, method, targetE, ifCheat=False, isFrac=False, Exc="x", addNoise=None):
    if method == "DFT":
        Exc = Exc
    else:
        Exc = None
    sys, ham, args, opt = initialGuess.setup_system(basis, method, file='test/water_equilim.xyz', 
                                           Exc=Exc, ifCheat=ifCheat, isFrac=isFrac)
    if addNoise is not None:
        args = gen_noise(addNoise, *args)
    cons = initialGuess.setup_cons(sys, args, N=opt["N"], N2=opt["N2"])
    
    lg,x0 = initialGuess.setup_lg(sys, ham, cons, args, basis, method, isFrac=isFrac)
    x_star,nIter = newton_rewrite.solve(lg, x0)
    
    initialGuess.check_E(ham, targetE)

def gen_noise(addNoise, *args):
    result = []
    for i in args:
        result.append(i+(addNoise*np.random.rand(*i.shape)))
    return result
    
#     ####TESTING
#     try:
#         sys.lf.disable_dual()
#     except AttributeError:
#         pass
#     #TESTING####
    
def frac_target_h2o_calc(basis, method, targetE, ifCheat=False, isFrac=False):
    if method == "DFT":
        Exc = "x"
    else:
        Exc = None
    sys, ham, args, opt = initialGuess.setup_system(basis, method, file='test/water_equilim.xyz', 
                                           Exc=Exc, ifCheat=ifCheat, isFrac=isFrac)
    
    sys.wfn.exp_alpha.occupations[4] = 0.5
    sys.wfn.exp_beta.occupations[4] = 0.5
    
    dm_a = sys.lf.create_one_body(sys.wfn.nbasis) 
    sys.wfn.exp_alpha.compute_density_matrix(dm_a)
    dm_b = sys.lf.create_one_body(sys.wfn.nbasis) 
    sys.wfn.exp_beta.compute_density_matrix(dm_b)
    
    sys.wfn.clear()
    setup_wfn(sys, ham, dm_a._array, dm_b._array)
    frac_args, N, N2 = initialGuess.setup_guess(sys, (sys.wfn.exp_alpha._coeffs, sys.wfn.exp_alpha.occupations,
                                          sys.wfn.exp_alpha.energies, sys.wfn.exp_beta._coeffs, 
                                          sys.wfn.exp_beta.occupations, sys.wfn.exp_beta.energies),
                                  isFrac=isFrac)
    
    cons = initialGuess.setup_cons(sys, args, N=4.5, N2=4.5)
    lg,x0 = initialGuess.setup_lg(sys, ham, cons, args, basis, method, isFrac=isFrac)
    x_star = solver.solve(lg, x0)
    initialGuess.check_E(ham, targetE)

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
default_h2o_calc('sto-3g', "DFT", -74.0689451960385, ifCheat=True, isFrac=True) #HORTON
# default_h2o_calc('6-31++g**', "DFT", -67.521923845983, ifCheat=True, isFrac=True) #NWCHEM
# default_h2o_calc('cc-pvqz', "DFT", -67.521923845983, ifCheat=True, isFrac=True) #NWCHEM

# projected_h2o_calc('3-21G', '6-31++G**', "DFT", -67.9894175486548, ifCheat=True, isFrac=True) #Horton
# frac_target_h2o_calc('sto-3g', "DFT", -66.634688718437, ifCheat=True, isFrac=True) #NWCHEM
