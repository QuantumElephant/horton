from horton.newton_rewrite import NewtonKrylov
from horton import *
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

def test_numerical_linear_constraint():
    xyzFile = context.get_fn('test/water_equilim.xyz')
    basis = 'sto-3g'
    system = System.from_file(xyzFile, obasis=basis)
    system.init_wfn(charge=0, restricted=False)
    guess_hamiltonian_core(system)
    
    grid = BeckeMolGrid(system, random_rotate=False, atspecs = 'tv-13.1-5')
    
    potential = system.compute_grid_orbitals(grid.points, iorbs=np.array([0]), select='alpha')
    
    potential = np.ones_like(potential)
    
    term = CustomGridFixedTerm(grid, potential.squeeze(), "L")
    operator = term.get_operator(system)[0]._array
    
    assert (np.abs(system.get_overlap()._array - operator) < 1e-5).all()
    
test_numerical_linear_constraint() 