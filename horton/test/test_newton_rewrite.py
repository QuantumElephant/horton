from horton import *
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold = 2000)

def calc_H2O():
    solver = NewtonKrylov()
#    basis = '3-21G'
#    basis = '6-31++G**'
    system = System.from_file(context.get_fn('test/H2_e.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
#    system = System.from_file(context.get_fn('test/water_sto3g_hf_g03.fchk'),context.get_fn('test/water_sto3g_hf_g03.log'),obasis='STO-3G')
#    system = System.from_file(context.get_fn('test/water_sto3g_hf_g03.fchk'),obasis='STO-3G')
    
##force open shell! 
#    system._wfn = None
#    system.init_wfn(restricted=False)
#    guess_hamiltonian_core(system)

    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    
    occ_a = np.array([0.5,0.5]); N=1; #STO-3G ONLY
    occ_b = np.array([0.5,0.5]); N2=1 #STO-3G ONLY
    
#     occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5; #STO-3G ONLY
#     occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    
#    occ_a = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N=5 #3-21G ONLY
#    occ_b = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N2=5 #3-21G ONLY

#    occ_a = np.array([1,0.5, 0.5,1/6.,1/6.,1/6.,1/6.,1/6.,1/6.,0.5, 0.5, 0.5 ,0.5]); N=5; print "Using atomic constrained occ" #3-21G ONLY
#    occ_b = np.array([1,0.5, 0.5,1/6.,1/6.,1/6.,1/6.,1/6.,1/6.,0.5, 0.5, 0.5 ,0.5]); N2=5; print "Using atomic constrained occ" #3-21G ONLY

    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)

#DFT
#    grid = BeckeMolGrid(system, random_rotate=False)
#    
#    libxc_term = LibXCLDATerm('c_vwn')
#    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

#HF
    ham = Hamiltonian(system, [HartreeFock()])

    L1_a0 = np.array(0.5)
    L1_b0 = np.array(0.5)
    L2_a0 = np.array(0.5)
    L2_b0 = np.array(0.5)
    L3_a0 = np.array(0.5)
    L3_b0 = np.array(0.5)
    Q1_0 = np.array(1)

#    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, L1_a0, L1_b0, L2_a0, L2_b0, L3_a0, L3_b0]
#    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, L1_a0, L1_b0, L2_a0, L2_b0, L3_a0, L3_b0]
#    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, L1_a0, L1_b0, Q1_0, Q1_0]
#    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, L1_a0, L1_b0, L2_b0]
#     args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub]
#    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, L1_a0, L1_b0, L2_a0]
#     args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, L1_a0, L2_a0, Q1_0]
    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, Q1_0]
#     args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, L1_a0]
#    args = [pro_da, pro_db, pro_ba, pro_bb, mua, mub] #Integer occupations
#    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, L1_a0, L1_b0]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    L1 = np.eye(dm_a.shape[0]);
    L2 = np.eye(dm_a.shape[0]);
    L3 = np.eye(dm_a.shape[0]);
    
    #Generate atomic occupancy matrix (L)
    prev_idx = 0
#     for key,i in enumerate([L1, L2, L3]): #WATER ONLY
    for key,i in enumerate([L1, L2]): #H2

        upper_R = prev_idx
        lower_R = nbasis[key] + prev_idx
        
        i[:upper_R, :upper_R] = 0
        i[lower_R:, lower_R:] = 0
        
        prev_idx += nbasis[key]
    
    L1_a = LinearConstraint(system, 1, L1) #WATER ONLY
    L2_a = LinearConstraint(system, 1, L2)
#     L3_a = LinearConstraint(system, 0.5, L3)
    
    L1_b = LinearConstraint(system, 4, L1) #WATER ONLY
    L2_b = LinearConstraint(system, .5, L2)
#     L3_b = LinearConstraint(system, 0.5, L3)
    
    Q1 = QuadraticConstraint(system, 0, [L1, L2])
    Q2 = QuadraticConstraint(system, 0, [L1, L2])
    
#    lg = Lagrangian(system, ham, [norm_a, L1_a, L2_a, L3_a,norm_b, L1_b, L2_b, L3_b])
#    lg = Lagrangian(system, ham,[L1_a, L2_a, L3_a,L1_b, L2_b, L3_b])
#    lg = Lagrangian(system, ham, [norm_a, L1_a, Q1,norm_b, L1_b, Q1])
#    lg = Lagrangian(system, ham, [norm_a, L1_a,norm_b, L1_b, L2_b])
#     lg = Lagrangian(system, ham,[norm_a,norm_b])
#    lg = Lagrangian(system, ham, [norm_a, L1_a, norm_b, L1_b, L2_a])
#     lg = Lagrangian(system, ham, [norm_a,norm_b, L1_a,L2_a,Q1])
#     lg = Lagrangian(system, ham, [norm_a, norm_b, Q1])
    lg = Lagrangian(system, ham,[norm_a, norm_b, Q1], isFrac = True) 
#     lg = Lagrangian(system, ham,[norm_a, L1_a, norm_b, L1_b]]) #O:4/4
    
    x0 = initialGuess.prep_D(lg, *args)

    lg.callback_system(x0, None)

    x_star = solver.solve(lg, x0)
    print "The Energy is " + str(ham.compute_energy())
    lg.calc_occupations(x_star)
    
    print lg.matHelper.vecToMat(x_star)
#     np.savez("UT_xstar", lg.matHelper.vecToMat(x_star))
#     np.savetxt("jacobianFinished", lg.fdiff_hess_grad_x(x_star))
    
    system._wfn = None
    system.init_wfn(restricted=False)
    ham.invalidate()
    guess_hamiltonian_core(system)
    
    converged = converge_scf_oda(ham)
    print ham.compute_energy() - system.compute_nucnuc()

    print "energy assertion deferred"
#    assert (abs(lg.wrap_callback_spin(x_star) - ham.compute_energy()) < 1e-4).all() #UPDATE

    
#    lg.occ_hist_a = np.vstack(lg.occ_hist_a)
#    lg.occ_hist_b = np.vstack(lg.occ_hist_b)
    lg.e_hist = np.vstack(lg.e_hist)
#    lg.e_hist_b = np.vstack(lg.e_hist_b)
    iters = np.arange(lg.nIter)

    plt.plot(iters[:-1],np.log10(np.abs(lg.e_hist[:-1] - lg.e_hist[-1])), 'r--')
    
    plt.show()
    
#def test_UTconvert():
#    basis = 'sto-3g' #CHANGE1
#    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
#    system.init_wfn(charge=0, mult=1, restricted=False)
#    
#    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b = initialGuess.promol_orbitals(system, basis)
#    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b)
#    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)
#    
#    ham = Hamiltonian(system, [HartreeFock()])
#
#    lf, lg = _lg_init(system, ham, N,N2,[pro_da, pro_db, pro_ba, pro_bb, pro_da, pro_db, mua, mub])
#    
#    ind = np.triu_indices(dm_a.shape[0])
#    
#    x0 = 2*np.hstack([pro_da[ind], pro_db[ind], pro_ba[ind], pro_bb[ind], pa[ind], pb[ind], 0.5*mua.squeeze(), 0.5*mub.squeeze()]) 
#    x0 = np.arange(x0.size)
#    
#    xOrig = cp.deepcopy(x0)
#    
#    a = lg.UTvecToMat(x0)
#    b = lg.UTmatToVec(*a)
#    
#    assert (np.abs(b - xOrig) < 1e-10).all(), np.abs(b - xOrig)
#    

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
    

def test_HF_STO3G():
    solver = NewtonKrylov()
#    
    basis = 'sto-3g'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)

#HF
    ham = Hamiltonian(system, [HartreeFock()])

    args = [pro_da, pro_ba, pro_db, pro_bb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = False)
    
    x0 = initialGuess.prep_D(lg, *args)

    print "start HF_STO3G"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-74.965901) #NIST
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -74.965901) < 1e-4
    


def test_HF_STO3G_H2_4():
    solver = NewtonKrylov()
#    
    basis = 'sto-3g'
    system = System.from_file(context.get_fn('test/H2_4.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([0.5,0.5]); N=1 #STO-3G ONLY
    occ_b = np.array([0.5,0.5]); N2=1 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)

#HF
    ham = Hamiltonian(system, [HartreeFock()])

    args = [pro_da, pro_db, pro_ba, pro_bb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = False)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "start HF_STO3G"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-74.965901) #NIST
    print "Computed E:" + str(ham.compute_energy())
#    assert np.abs(ham.compute_energy() - -74.965901) < 1e-4
    

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
#     args = [pro_da, pro_db, pro_ba, pro_bb, mua, mub]
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
    

def test_DFT_STO3G_Frac_H2_4():
    solver = NewtonKrylov()
#    
    basis = 'sto-3g'
    system = System.from_file(context.get_fn('test/H2_4.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis, np.array([[1,-1,1]]))
    occ_a = np.array([0.5,0.5]); N=1 #STO-3G ONLY
    occ_b = np.array([0.5,0.5]); N2=1 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)

    grid = BeckeMolGrid(system, random_rotate=False)
    
    libxc_term = LibXCLDATerm('c_vwn') #vwn_5
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "Start DFT_STO3G_Frac"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-66.634688718437) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
#    assert np.abs(ham.compute_energy() - -66.634688718437) < 1e-4
    

def test_HF_321G():
    solver = NewtonKrylov()
#    
    basis = '3-21G'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N=5 #3-21G ONLY
    occ_b = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N2=5 #3-21G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
#HF
    ham = Hamiltonian(system, [HartreeFock()])

    args = [pro_da, pro_db, pro_ba, pro_bb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = False)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "start HF_321G"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-75.583747447860) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -75.583747447860) < 1e-4 #KNOWN TO FAIL
    


def test_HF_631G():
    solver = NewtonKrylov()
#    
    basis = '6-31++G**'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
#    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
#    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    N = 5; N2 = 5;
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
#HF
    ham = Hamiltonian(system, [HartreeFock()])

    args = [pro_da, pro_db, pro_ba, pro_bb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")


    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = False)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "start HF_631G"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-74.965901) #NIST
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -74.965901) < 1e-4
    

def test_DFT_STO3G():
    solver = NewtonKrylov()
#    
    basis = 'sto-3g'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)

    grid = BeckeMolGrid(system, random_rotate=False)
    
    libxc_term = LibXCLDATerm('c_vwn') #vwn_5
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

    args = [pro_da, pro_ba, pro_db, pro_bb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = False)
    
    x0 = initialGuess.prep_D(lg, *args)

    print "start DFT_STO3G"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-66.634688718437) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -66.634688718437) < 1e-4
    

def test_DFT_321G():
    solver = NewtonKrylov()
#    
    basis = '3-21G'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N=5 #3-21G ONLY
    occ_b = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N2=5 #3-21G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)

    grid = BeckeMolGrid(system, random_rotate=False)
    
    libxc_term = LibXCLDATerm('c_vwn') #vwn_5
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

    args = [pro_da, pro_db, pro_ba, pro_bb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = False)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "start DFT_321G"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-67.521923845983) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -67.521923845983) < 1e-4 #KNOWN TO FAIL
    

def test_HF_STO3G_Frac():
    solver = NewtonKrylov()
#    
    basis = 'sto-3g'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)

#HF
    ham = Hamiltonian(system, [HartreeFock()])

    args = [pro_da, pro_ba, pa, pro_db, pro_bb, pb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
    
    x0 = initialGuess.prep_D(lg, *args) 
    lg.energy_wrap(x0)

    print "start HF_STO3G_Frac"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-74.965901)
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -74.965901) < 1e-4 #KNOWN TO FAIL -74.5638326142
    

def test_HF_321G_Frac():
    solver = NewtonKrylov()
    
    basis = '3-21G'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N=5 #3-21G ONLY
    occ_b = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N2=5 #3-21G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)

#HF
    ham = Hamiltonian(system, [HartreeFock()])

    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "Start HF_321G_Frac"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-75.0812082641)
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -75.0812082641) < 1e-10 #KNOWN TO FAIL -75.3159038895
    

def test_DFT_STO3G_Frac():
    solver = NewtonKrylov()
#    
    basis = 'sto-3g'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N=5 #STO-3G ONLY
    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]); N2=5 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)

    grid = BeckeMolGrid(system, random_rotate=False)
    
    libxc_term = LibXCLDATerm('c_vwn') #vwn_5
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "Start DFT_STO3G_Frac"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-66.634688718437) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -66.634688718437) < 1e-4
    

def test_DFT_321G_Frac():
    solver = NewtonKrylov()
#    
    basis = '3-21G'
    system = System.from_file(context.get_fn('test/water_equilim.xyz'), obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N=5 #3-21G ONLY
    occ_b = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N2=5 #3-21G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)

    grid = BeckeMolGrid(system, random_rotate=False)
    
    libxc_term = LibXCLDATerm('c_vwn') #vwn_5
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub]

    norm_a = LinearConstraint(system, N, np.eye(dm_a.shape[0]), select="alpha")
    norm_b = LinearConstraint(system, N2, np.eye(dm_a.shape[0]), select="beta")

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "start DFT_321G_Frac"
    x_star = solver.solve(lg, x0)
    
    print "Actual E:" + str(-67.521923845983) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
    assert np.abs(ham.compute_energy() - -67.521923845983) < 1e-4 #KNOWN TO FAIL

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
    norm_b = LinearConstraint(system, N2-0.5, np.eye(dm_a.shape[0]), select="beta", C_init=N, steps=50)

    lg = Lagrangian(system, ham, [norm_a, norm_b], isFrac = True)
    
    x0 = initialGuess.prep_D(lg, *args) 

    print "Start DFT_STO3G_Frac"
    x_star = solver.fancy_solve(lg, x0)
    
    print "Actual E:" + str(73.6549343469) #NWCHEM
    print "Computed E:" + str(ham.compute_energy())
#     assert np.abs(ham.compute_energy() - -73.6549343469) < 1e-4

def test_quadratic_stepped_constraints():
    solver = NewtonKrylov()
    basis = "sto-3g"
    sys, ham = initialGuess.generic_DFT_calc(basis = basis, lda_term = "c_vwn")
    
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
    
  
# test_linear_stepped_constraints()
test_quadratic_stepped_constraints()
# # calc_H2O()
# # Horton_H2O()
# test_HF_STO3G()
# test_HF_STO3G_H2_4()
# # test_DFT_STO3G_H2_4()
# test_DFT_STO3G_Frac_H2_4()
# test_HF_321G()
# test_HF_631G()
# test_DFT_STO3G()
# test_DFT_321G()
# test_HF_STO3G_Frac()
# test_HF_321G_Frac()
# # test_DFT_STO3G_Frac()
# # test_DFT_321G_Frac()
