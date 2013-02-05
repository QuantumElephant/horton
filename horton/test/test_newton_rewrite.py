from horton.newton_rewrite import NewtonKrylov
from horton import *
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

np.set_printoptions(threshold = 2000)

def get_water_sto3g_hf(lf=None):
    if lf is None:
        lf = DenseLinalgFactory()
    fn = context.get_fn('test/water_sto3g_hf_g03.log')
    overlap, kinetic, nuclear_attraction, electronic_repulsion = load_operators_g09(fn, lf)
    coeffs = np.array([
        9.94099882E-01, 2.67799213E-02, 3.46630004E-03, -1.54676269E-15,
        2.45105601E-03, -6.08393842E-03, -6.08393693E-03, -2.32889095E-01,
        8.31788042E-01, 1.03349385E-01, 9.97532839E-17, 7.30794097E-02,
        1.60223990E-01, 1.60223948E-01, 1.65502862E-08, -9.03020258E-08,
        -3.46565859E-01, -2.28559667E-16, 4.90116062E-01, 4.41542336E-01,
        -4.41542341E-01, 1.00235366E-01, -5.23423149E-01, 6.48259144E-01,
        -5.78009326E-16, 4.58390414E-01, 2.69085788E-01, 2.69085849E-01,
        8.92936017E-17, -1.75482465E-16, 2.47517845E-16, 1.00000000E+00,
        5.97439610E-16, -3.70474007E-17, -2.27323914E-17, -1.35631600E-01,
        9.08581133E-01, 5.83295647E-01, -4.37819173E-16, 4.12453695E-01,
        -8.07337352E-01, -8.07337875E-01, 5.67656309E-08, -4.29452066E-07,
        5.82525068E-01, -6.76605679E-17, -8.23811720E-01, 8.42614916E-01,
        -8.42614243E-01
    ]).reshape(7,7).T
    epsilons = np.array([
        -2.02333942E+01, -1.26583942E+00, -6.29365088E-01, -4.41724988E-01,
        -3.87671783E-01, 6.03082408E-01, 7.66134805E-01
    ])
#    wfn = ClosedShellWFN(lf=lf, nbasis=7)
#    wfn.expansion.coeffs[:] = coeffs
#    wfn.expansion.energies[:] = epsilons
    wfn = None
    return lf, overlap, kinetic, nuclear_attraction, electronic_repulsion, wfn

def _lg_init(sys, ham, N,  N2, x0_sample, ee_rep=None):
    lf = sys._lf
    overlap = sys.get_overlap()
    kinetic = sys.get_kinetic()
    nuclear_attraction = sys.get_nuclear_attraction()
    electronic_repulsion = sys.get_electron_repulsion()
#    lf = DenseLinalgFactory()
#    overlap, kinetic, nuclear_attraction, electronic_repulsion = load_operators_g09(context, lf)
    nbasis = overlap.nbasis

    if ee_rep is not None:
        if isinstance(ee_rep,np.ndarray):
            nbasis = ee_rep.shape[0]
            electronic_repulsion = lf.create_two_body(nbasis)
            electronic_repulsion._array = ee_rep
        else:
            nbasis = ee_rep
            electronic_repulsion._array = electronic_repulsion._array[0:nbasis, 0:nbasis, 0:nbasis, 0:nbasis]

    Vee = HF_dVee(electronic_repulsion, nbasis, lf)
    W = []

    lg = lagrangian(lf, kinetic, nuclear_attraction, overlap, N, N2, Vee, W, x0_sample, sys, ham) # TODO: check normalization

    return lf, lg, nbasis, wfn

def _lg_init_context(context, N,  N2, x0_sample, ee_rep=None):
#    lf = sys._lf
#    overlap = sys.get_overlap()
#    kinetic = sys.get_kinetic()
#    nuclear_attraction = sys.get_nuclear_attraction()
#    electronic_repulsion = sys.get_electron_repulsion()
    lf = DenseLinalgFactory()
    overlap, kinetic, nuclear_attraction, electronic_repulsion = load_operators_g09(context, lf)
    nbasis = overlap.nbasis

    if ee_rep is not None:
        if isinstance(ee_rep,np.ndarray):
            nbasis = ee_rep.shape[0]
            electronic_repulsion = lf.create_two_body(nbasis)
            electronic_repulsion._array = ee_rep
        else:
            nbasis = ee_rep
            electronic_repulsion._array = electronic_repulsion._array[0:nbasis, 0:nbasis, 0:nbasis, 0:nbasis]

    Vee = HF_dVee(electronic_repulsion, nbasis, lf)
    W = []

    lg = lagrangian(lf, kinetic, nuclear_attraction, overlap, N, N2, Vee, W, x0_sample) # TODO: check normalization

    return lf, lg, nbasis, wfn



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
#        int1d = SimpsonIntegrator1D()
#        rtf = ExpRTransform(1e-3, 10.0, 100)
#        grid = BeckeMolGrid(atom_sys, (rtf, int1d, 110), random_rotate=False)
#        
#        libxc_term = LibXCLDATerm('x')
#        ham = Hamiltonian(atom_sys, [Hartree(), libxc_term], grid)

#HF
        ham = Hamiltonian(atom_sys, [HartreeFock()])
        
        converge_scf_oda(ham)
        
        orb_alpha.append(atom_sys.wfn.exp_alpha._coeffs) #Why is the sign wrong?
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
#    energy_a = np.array([-20.2767318,-1.06331854,-0.392195929,-0.057694873,-0.057694873,-0.0528525855,-0.0528525855]) #ROHF
#    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.5,0.5]);
#    orb_a = np.array([[0.995191325,0.0195527548,0,0,0,0,0],
#                        [-0.2625805,1.02906364,0,0,0,0,0],
#                        [0,0,0,0,1,0,0],
#                        [0,0,0,1,0,0,0],
#                        [0,0,1,0,0,0,0],
#                        [0,0,0,0,0,1,0],
#                        [0,0,0,0,0,0,1]])
#    
#    energy_a = np.array([-20.31123,-1.29309,-0.55034,-0.55034,-0.45546,-0.46658185,-0.46658185]) #UHF alpha
#    occ_a = np.array([1,1,2/3.,2/3.,2/3.,0.375,0.375]); N=4.75
    occ_a = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N=5
#    orb_a = np.array([[0.99453,-0.26507,0,0,0,0,0],
#                        [0.02213,1.02901,0,0,0,0,0],
#                        [0,0,0,1,0,0,0],
#                        [0,0,1,0,0,0,0],
#                        [0,0,0,0,1,0,0],
#                        [0,0,0,0,0,1,0],
#                        [0,0,0,0,0,0,1]])
###    
#    energy_b = np.array([-20.25957,-0.94834,-0.36057,0.37726,0.37726,0.308024094,0.308024094]) #UHF beta
#    occ_b = np.array([1,1,2/3.,2/3.,2/3.,0.375,0.375]); N2 = 4.75
    occ_b = np.array([1,0.5, 0.5,2/6.,2/6.,2/6.,2/6.,2/6.,2/6.,0.25, 0.25, 0.25 ,0.25]); N2=5
#    orb_b = np.array([[0.99551,-0.26135,0,0,0,0,0],
#                        [0.01828,1.02909,0,0,0,0,0],
#                        [0,0,0,0,1,0,0],
#                        [0,0,0,1,0,0,0],
#                        [0,0,1,0,0,0,0],
#                        [0,0,0,0,0,1,0],
#                        [0,0,0,0,0,0,1]])
    
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
    
#    mua = np.ones([1])*-0.45546
#    mub = np.ones([1])*-0.45546
    mua = np.ones([1])*np.max(energy_a[energy_a<0]) #hack. Can't handle size 1 arrays cleanly.
    mub = np.ones([1])*np.max(energy_a[energy_a<0])
    
#    return [pro_da, pro_ba, cp.deepcopy(pro_da), cp.deepcopy(pro_ba), mua, mub, N, N2]
    return [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2]


def test_H2O_DM():
    solver = NewtonKrylov()
    [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2] = promol_h2o()
    lf, lg, nbasis, wfn = _lg_init(get_water_sto3g_hf, N,N2,[pro_da, pro_db, pro_ba, pro_bb, pro_da, pro_db, mua, mub])
    
    S = lg.S
    
    pa = sqrtm(np.dot(np.dot(S,pro_da),S) - np.dot(np.dot(np.dot(np.dot(S,pro_da),S),pro_da),S))
    pb = sqrtm(np.dot(np.dot(S,pro_db),S) - np.dot(np.dot(np.dot(np.dot(S,pro_db),S),pro_db),S))
    
    x0 = np.hstack([pro_da.ravel(), pro_db.ravel(), pro_ba.ravel(), pro_bb.ravel(), pa.ravel(), pb.ravel(), mua, mub])

    x_star = solver.solve(lg, x0)

    assert (abs(lg.wrap_callback_spin(x_star) + 84.212859533) < 1e-4).all()
    
#test_H2O_DM()


def test_H2O_New():
    solver = NewtonKrylov()
#    
    basis = 'sto-3g' #CHANGE1
#    basis = '3-21G'
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
#    int1d = SimpsonIntegrator1D()
#    rtf = ExpRTransform(1e-3, 10.0, 100)
#    grid = BeckeMolGrid(system, (rtf, int1d, 110), random_rotate=False)
    
#    grid = BeckeMolGrid(system, random_rotate=False)
##    
#    libxc_term = LibXCLDATerm('x')
#    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)

#HF
    ham = Hamiltonian(system, [HartreeFock()])

    lf, lg, nbasis, wfn = _lg_init(system, ham, N,N2,[pro_da, pro_db, pro_ba, pro_bb, pro_da, pro_db, mua, mub])
    
    pa = sqrtm(np.dot(np.dot(S,pro_da),S) - np.dot(np.dot(np.dot(np.dot(S,pro_da),S),pro_da),S))
    pb = sqrtm(np.dot(np.dot(S,pro_db),S) - np.dot(np.dot(np.dot(np.dot(S,pro_db),S),pro_db),S))
    
    ind = np.triu_indices(dm_a.shape[0])
    
    x0 = np.hstack([pro_da.ravel(), pro_db.ravel(), pro_ba.ravel(), pro_bb.ravel(), pa.ravel(), pb.ravel(), mua, mub]); lg.isUT = False
#    x0 = np.hstack([pro_da[ind], pro_db[ind], pro_ba[ind], pro_bb[ind], pa[ind], pb[ind], mua, mub]); lg.isUT = True
#    lg.test_UTconvert(x0)

    x_star = solver.solve(lg, x0)
    
    if lg.isUT:
        print lg.UTvecToMat(x_star)
        np.savetxt("jacobianFinished", lg.fdiff_hess_slow(lg.sym_gradient_spin_frac, *lg.UTvecToMat(x_star)))
    else:
        print lg.vecToMat(x_star)
        np.savetxt("jacobianFinished", lg.fdiff_hess_slow(lg.gradient_spin_frac, *lg.vecToMat(x_star)))
    
    system._wfn = None
    system.init_wfn(restricted=False)
    ham.invalidate()
    guess_hamiltonian_core(system)
    
    converged = converge_scf_oda(ham)
#    print ham.compute_energy() - system.compute_nucnuc()

    print "energy assertion deferred"
#    assert (abs(lg.wrap_callback_spin(x_star) - ham.compute_energy()) < 1e-4).all()
#    assert (abs(lg.wrap_callback_spin(x_star) + 84.212859533) < 1e-4).all()

    
#    lg.occ_hist_a = np.vstack(lg.occ_hist_a)
#    lg.occ_hist_b = np.vstack(lg.occ_hist_b)
    lg.e_hist_a = np.vstack(lg.e_hist_a)
#    lg.e_hist_b = np.vstack(lg.e_hist_b)
    iters = np.arange(lg.nIter)
#    fig = plt.figure()
#    ax1 = fig.add_subplot(221)
#    ax2 = fig.add_subplot(212)
    
#    for i in np.arange(lg.occ_hist_a.shape[1]):
#        ax1.plot(iters,lg.occ_hist_a[:,i], 'r--', iters, lg.occ_hist_b[:,i], 'b--') 
    plt.plot(iters,np.log10(np.abs(lg.e_hist_a - -74.4141088197)), 'r--')
    
    plt.show()
    
test_H2O_New()

def test_Horton_H2O():
    basis = '3-21G'
    system = System.from_file('water_equilim.xyz', obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    guess_hamiltonian_core(system)
#    
#    int1d = SimpsonIntegrator1D()
#    rtf = ExpRTransform(1e-3, 10.0, 100)
#    grid = BeckeMolGrid(system, (rtf, int1d, 110), random_rotate=False)

    grid = BeckeMolGrid(system, random_rotate=False)
    
    libxc_term = LibXCLDATerm('x')
    ham = Hamiltonian(system, [Hartree(), libxc_term], grid)
    
    converged = converge_scf_oda(ham, max_iter=5000)
    
#test_Horton_H2O()

def test_fdiff_hess_slow():
    basis = '3-21G'
    system = System.from_file('water_equilim.xyz', obasis=basis)
    system.init_wfn(charge=0, mult=1, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b = promol_guess(system, basis)
    [pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2] = promol_h2o(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b)

    S = system.get_overlap()._array
    
    ham = Hamiltonian(system, [HartreeFock()])

    lf, lg, nbasis, wfn = _lg_init(system, ham, N,N2,[pro_da, pro_db, pro_ba, pro_bb, pro_da, pro_db, mua, mub])
    
    pa = sqrtm(np.dot(np.dot(S,pro_da),S) - np.dot(np.dot(np.dot(np.dot(S,pro_da),S),pro_da),S))
    pb = sqrtm(np.dot(np.dot(S,pro_db),S) - np.dot(np.dot(np.dot(np.dot(S,pro_db),S),pro_db),S))
        
    x0 = np.hstack([pro_da.ravel(), pro_db.ravel(), pro_ba.ravel(), pro_bb.ravel(), pa.ravel(), pb.ravel(), mua, mub])
    
    print "HESSIAN" 
#    a = lg.fdiff_gradient(*lg.lin_wrap(x0))
    b = lg.fdiff_hess_slow(*lg.lin_wrap(x0))
    
#    for key,i in enumerate(a):
#        print np.linalg.norm(a[key] - b[key])
    
    np.savetxt("jacobian", b)
    
#test_fdiff_hess_slow()