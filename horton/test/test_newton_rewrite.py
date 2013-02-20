from horton.newton_rewrite import NewtonKrylov
from horton import *
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

np.set_printoptions(threshold = 2000)

#def _lg_init(sys, ham, N,  N2, x0_sample, ee_rep=None):
#    lf = sys._lf
#    overlap = sys.get_overlap()
#    lg = lagrangian(lf, overlap, N, N2, x0_sample, sys, ham) # TODO: check normalization
#    return lf, lg

def sqrtm(A):
    result = np.zeros_like(A)
    eigvs,eigvc = np.linalg.eigh(A)
    for i in np.arange(len(eigvs)):
        if eigvs[i] > 0:
            result += np.sqrt(eigvs[i])*np.outer(eigvc[i], eigvc[i])
    
    return result

def promol_guess(sys, basis):
    orb_alpha = []
    orb_beta = []
    occ_alpha = []
    occ_beta = []
    e_alpha = []
    e_beta = []
    
    for i in np.sort(sys.numbers)[::-1]:
        atom_sys = System(np.zeros((1,3), float), np.array([i]), obasis=basis) #hacky
        if i > 1:
            atom_sys.init_wfn(charge=0, mult=3, restricted=False)
        else:
            atom_sys.init_wfn(restricted=False)        

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

#    occ_alpha = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5])
#    occ_beta = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5])
    
    e_alpha = np.hstack(e_alpha)
    e_beta = np.hstack(e_beta)
    
    return orb_alpha, orb_beta, occ_alpha, occ_beta, e_alpha, e_beta
    

def promol_h2o(orb_a, orb_b, occ_a, occ_b, energy_a, energy_b):
    occ_a = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N=5
    occ_b = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N2=5
    
#    N = np.sum(occ_a)
#    N2 = np.sum(occ_b)
#    N2 = np.sum(occ_a)
    
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
    
    return [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2]


def test_grad():
    basis = 'sto-3g' #CHANGE1
    system = System.from_file('water_equilim.xyz', obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b = promol_guess(system, basis)
    [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2] = promol_h2o(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b)

    S = system.get_overlap()._array
    
    ham = Hamiltonian(system, [HartreeFock()])

    lg = Lagrangian(system, ham, [pro_da, pro_db, pro_ba, pro_bb, pro_da, pro_db, mua, mub])
    
    pa = sqrtm(np.dot(np.dot(S,pro_da),S) - np.dot(np.dot(np.dot(np.dot(S,pro_da),S),pro_da),S)) #Move to promol_guess()
    pb = sqrtm(np.dot(np.dot(S,pro_db),S) - np.dot(np.dot(np.dot(np.dot(S,pro_db),S),pro_db),S)) #Move to promol_guess()
    
    x0 = np.hstack([pro_da.ravel(), pro_db.ravel(), pro_ba.ravel(), pro_bb.ravel(), pa.ravel(), pb.ravel(), mua, mub]); lg.isUT = False

    assert np.abs(lg.fdiff_hess_grad_grad(x0) - lg.lin_grad_wrap(x0)) < 1e-6

def test_UTgrad():
    basis = 'sto-3g' #CHANGE1
    system = System.from_file('water_equilim.xyz', obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b = promol_guess(system, basis)
    [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2] = promol_h2o(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b)

    S = system.get_overlap()._array
    
    ham = Hamiltonian(system, [HartreeFock()])

    lg = Lagrangian(system, ham, [pro_da, pro_db, pro_ba, pro_bb, pro_da, pro_db, mua, mub])
    
    pa = sqrtm(np.dot(np.dot(S,pro_da),S) - np.dot(np.dot(np.dot(np.dot(S,pro_da),S),pro_da),S)) #Move to promol_guess()
    pb = sqrtm(np.dot(np.dot(S,pro_db),S) - np.dot(np.dot(np.dot(np.dot(S,pro_db),S),pro_db),S)) #Move to promol_guess()
    
    x0 = prep_D(pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub); lg.isUT = True; lg.tri_offsets()

    assert np.abs(lg.fdiff_hess_grad_grad(x0) - lg.lin_grad_wrap(x0)) < 1e-6
    
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

def test_H2O():
    solver = NewtonKrylov()
#    
#    basis = 'sto-3g'
    basis = '3-21G'
    system = System.from_file('water_equilim.xyz', obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
#    system = System.from_file(context.get_fn('test/water_sto3g_hf_g03.fchk'),context.get_fn('test/water_sto3g_hf_g03.log'),obasis='STO-3G')
#    system = System.from_file(context.get_fn('test/water_sto3g_hf_g03.fchk'),obasis='STO-3G')
    
##force open shell! 
#    system._wfn = None
#    system.init_wfn(restricted=False)
#    guess_hamiltonian_core(system)

    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b = promol_guess(system, basis)
    [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2] = promol_h2o(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b)

    S = system.get_overlap()._array

#DFT
    grid = BeckeMolGrid(system, random_rotate=False)
    
    libxc_term = LibXCLDATerm('x')
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

#HF
#    ham = Hamiltonian(system, [HartreeFock()])

    L1_a0 = np.array(0.5)
    L1_b0 = np.array(0.5)
    L2_a0 = np.array(0.5)
    L2_b0 = np.array(0.5)
    L3_a0 = np.array(0.5)
    L3_b0 = np.array(0.5)

    pa = sqrtm(np.dot(np.dot(S,pro_da),S) - np.dot(np.dot(np.dot(np.dot(S,pro_da),S),pro_da),S)) #Move to promol_guess()
    pb = sqrtm(np.dot(np.dot(S,pro_db),S) - np.dot(np.dot(np.dot(np.dot(S,pro_db),S),pro_db),S)) #Move to promol_guess()

#    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub, L1_a0, L1_b0, L2_a0, L2_b0, L3_a0, L3_b0]
    args = [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua, mub]

    norm_a = Constraint(system, N, np.eye(dm_a.shape[0]))
    norm_b = Constraint(system, N2, np.eye(dm_a.shape[0]))

    L1 = np.eye(dm_a.shape[0]); print "3-21G ONLY!"
    L1[9:,9:] = 0; print "3-21G ONLY!"
    
    L2 = np.eye(dm_a.shape[0]); print "3-21G ONLY!"
    L2[:9,:9] = 0; print "3-21G ONLY!"
    L2[11:,11:] = 0; print "3-21G ONLY!"
    
    L3 = np.eye(dm_a.shape[0]); print "3-21G ONLY!"
    L3[:11,:11] = 0; print "3-21G ONLY!"
    
    
    L1_a = Constraint(system, 3, L1)
    L2_a = Constraint(system, 1, L2)
    L3_a = Constraint(system, 1, L3)
    
    L1_b = Constraint(system, 3, L1)
    L2_b = Constraint(system, 1, L2)
    L3_b = Constraint(system, 1, L3)

    shapes = []
    for i in args:
        if i.size == 1:
            shapes.append(i.size)
            continue
        shapes.append(i.shape[0])

#    lg = Lagrangian(system, ham,N, N2, shapes, [[norm_a, L1_a, L2_a, L3_a],[norm_b, L1_b, L2_b, L3_b]])
    lg = Lagrangian(system, ham,N, N2, shapes, [[norm_a],[norm_b]])
    
#    a = np.load("full_xstar.npz")
#    [pro_da, pro_db, pro_ba, pro_bb, pa, pb, mua,mub] = a["arr_0"]
    
#    x0 = np.hstack(*args); lg.isUT = False; lg.full_offsets()
    x0 = prep_D(*args); lg.isUT = True; lg.tri_offsets()

#    lg.fdiff_hess_grad_x(x0)

    x_star = solver.solve(lg, x0)
    
    if lg.isUT:
        print lg.UTvecToMat(x_star)
#        np.savez("UT_xstar", lg.vecToMat(x_star))
#        np.savetxt("jacobianFinished", lg.fdiff_hess_grad_x(x_star))
    else:
        print lg.vecToMat(x_star)
#        np.savez("full_xstar", lg.vecToMat(x_star))
#        np.savetxt("jacobianFinished", lg.fdiff_hess_grad_x(x_star))
    
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
    
test_H2O()

#def test_UTconvert():
#    basis = 'sto-3g' #CHANGE1
#    system = System.from_file('water_equilim.xyz', obasis=basis)
#    system.init_wfn(charge=0, mult=1, restricted=False)
#    
#    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b = promol_guess(system, basis)
#    [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2] = promol_h2o(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b)
#
#    S = system.get_overlap()._array
#    
#    ham = Hamiltonian(system, [HartreeFock()])
#
#    lf, lg = _lg_init(system, ham, N,N2,[pro_da, pro_db, pro_ba, pro_bb, pro_da, pro_db, mua, mub])
#    
#    pa = sqrtm(np.dot(np.dot(S,pro_da),S) - np.dot(np.dot(np.dot(np.dot(S,pro_da),S),pro_da),S)) #Move to promol_guess()
#    pb = sqrtm(np.dot(np.dot(S,pro_db),S) - np.dot(np.dot(np.dot(np.dot(S,pro_db),S),pro_db),S)) #Move to promol_guess()
#    
#    ind = np.triu_indices(dm_a.shape[0])
#    
#    x0 = 2*np.hstack([pro_da[ind], pro_db[ind], pro_ba[ind], pro_bb[ind], pa[ind], pb[ind], 0.5*mua.squeeze(), 0.5*mub.squeeze()]); lg.isUT = True; lg.tri_offsets()
#    x0 = np.arange(x0.size)
#    
#    xOrig = cp.deepcopy(x0)
#    
#    a = lg.UTvecToMat(x0)
#    b = lg.UTmatToVec(*a)
#    
#    assert (np.abs(b - xOrig) < 1e-10).all(), np.abs(b - xOrig)
#    
#test_UTconvert()

def Horton_H2O():
    basis = '3-21G'
    system = System.from_file('water_equilim.xyz', obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    guess_hamiltonian_core(system)
#  DFT  
#    grid = BeckeMolGrid(system, random_rotate=False)
#    libxc_term = LibXCLDATerm('x')
#    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
    
#  HF
    ham = Hamiltonian(system, [HartreeFock()])
    
    converged = converge_scf_oda(ham, max_iter=5000)
    
#Horton_H2O()

#def fdiff_hess_slow():
#    basis = '3-21G'
#    system = System.from_file('water_equilim.xyz', obasis=basis)
#    system.init_wfn(charge=0, mult=1, restricted=False)
#    
#    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b = promol_guess(system, basis)
#    [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2] = promol_h2o(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b)
#
#    S = system.get_overlap()._array
#    
#    ham = Hamiltonian(system, [HartreeFock()])
#
#    lf, lg = _lg_init(system, ham, N,N2,[pro_da, pro_db, pro_ba, pro_bb, pro_da, pro_db, mua, mub])
#    
#    pa = sqrtm(np.dot(np.dot(S,pro_da),S) - np.dot(np.dot(np.dot(np.dot(S,pro_da),S),pro_da),S))
#    pb = sqrtm(np.dot(np.dot(S,pro_db),S) - np.dot(np.dot(np.dot(np.dot(S,pro_db),S),pro_db),S))
#        
#    x0 = np.hstack([pro_da.ravel(), pro_db.ravel(), pro_ba.ravel(), pro_bb.ravel(), pa.ravel(), pb.ravel(), mua, mub])
#    
#    print "HESSIAN" 
##    a = lg.fdiff_gradient(*lg.lin_wrap(x0))
#    b = lg.fdiff_hess_grad_x(x0)
#    
##    for key,i in enumerate(a):
##        print np.linalg.norm(a[key] - b[key])
#    
#    np.savetxt("jacobian", b)
    
#fdiff_hess_slow()