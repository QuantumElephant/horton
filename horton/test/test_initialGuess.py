from horton.newton_rewrite import NewtonKrylov
from horton import *
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from horton.guess import guess_hamiltonian_core

def test_projection():
    xyzFile = context.get_fn('test/water_equilim.xyz')
    basis = 'sto-3g'
    system = System.from_file(xyzFile, obasis=basis)
    system.init_wfn(charge=0, restricted=False)
    
    dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, nbasis = initialGuess.promol_orbitals(system, basis)
    occ_a = np.array([1,1,1,0.5,0.5,0.5,0.5]); N=5 #STO-3G ONLY
    occ_b = np.array([1,1,1,0.5,0.5,0.5,0.5]); N2=5 #STO-3G ONLY
    pro_da, pro_ba, pro_db, pro_bb, mua, mub, N, N2 = initialGuess.promol_guess(dm_a, dm_b, occ_a, occ_b, energy_a, energy_b, N, N2)
    pa, pb = initialGuess.promol_frac(system, pro_da, pro_db)
    
    newBasis = '6-31++g**'
    system2 = System.from_file(xyzFile, obasis=newBasis)
    system2.init_wfn(charge=0, restricted=False)
    
    result = initialGuess.project(system, system2, pro_da, pro_ba, pro_db, pro_bb, pa, pb, mua, mub)
    print map(np.shape, result)
    
    old_occ = map(lambda x: np.trace(x.dot(system.get_overlap()._array)), [pro_da, pro_db])
    new_occ = map(lambda x: np.trace(x.dot(system2.get_overlap()._array)), [result[0], result[2]])
    
    for i,j in zip(old_occ, new_occ):
        assert np.abs(i-j) < 1e-2, (i,j)

def test_mcPurify():
    system, ham, basis = initialGuess.generic_DFT_calc()
    S = system.get_overlap()._array
    guess_hamiltonian_core(system)
    D = system.wfn.dm_alpha._array
    
    Eigval_before = np.linalg.eigvalsh(np.dot(S,D))
    avgEigval_before = np.average(np.abs(Eigval_before -0.5))
    
    D = initialGuess.mcPurify(system, D)[0]
    
    Eigval_after = np.linalg.eigvalsh(np.dot(S,D))
    avgEigval_after = np.average(np.abs(Eigval_after -0.5))
    
    assert avgEigval_after > avgEigval_before

def test_normalize_D():
    system, ham, basis = initialGuess.generic_DFT_calc()
    S = system.get_overlap()._array
    guess_hamiltonian_core(system)
    D = system.wfn.dm_alpha._array
    
    newD = normalize_D(system, D, 5)
    
    N = np.dot(system.get_overlap()._array.ravel(), newD.ravel())
    assert np.abs(N - 5) < 1e-8

# test_normalize_D()
# test_mcPurify()
# test_projection()

