from horton.newton_rewrite import NewtonKrylov
from horton import *
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

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
        
#     for i,j in ((pro_da,result[0]),(pro_db, result[1])):
#         ni = np.linalg.norm(i)
#         nj = np.linalg.norm(j)
#         assert np.abs(ni-nj)<1e-2, (ni,nj)

# test_projection()

