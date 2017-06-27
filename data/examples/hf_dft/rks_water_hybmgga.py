#!/usr/bin/env python
#JSON {"lot": "RKS/6-31G(d)",
#JSON  "scf": "CDIISSCFSolver",
#JSON  "linalg": "CholeskyLinalgFactory",
#JSON  "difficulty": 7,
#JSON  "description": "Basic RKS DFT example with hybrid MGGA exhange-correlation functional (TPSS)"}

from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import


# Load the coordinates from file.
# Use the XYZ file from HORTON's test data directory.
fn_xyz = context.get_fn('test/water.xyz')
mol = IOData.from_file(fn_xyz)

# Create a Gaussian basis set
obasis = get_gobasis(mol.coordinates, mol.numbers, '6-31g(d)')

# Create a linalg factory
lf = CholeskyLinalgFactory(obasis.nbasis)

# Compute Gaussian integrals
olp = obasis.compute_overlap(lf)
kin = obasis.compute_kinetic(lf)
na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
er = obasis.compute_electron_repulsion(lf)

# Define a numerical integration grid needed the XC functionals
grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers)

# Create alpha orbitals
exp_alpha = lf.create_expansion()

# Initial guess
guess_core_hamiltonian(olp, kin, na, exp_alpha)

# Construct the restricted HF effective Hamiltonian
external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
libxc_term = RLibXCHybridMGGA('xc_m05')
terms = [
    RTwoIndexTerm(kin, 'kin'),
    RDirectTerm(er, 'hartree'),
    RGridGroup(obasis, grid, [libxc_term]),
    RExchangeTerm(er, 'x_hf', libxc_term.get_exx_fraction()),
    RTwoIndexTerm(na, 'ne'),
]
ham = REffHam(terms, external)

# Decide how to occupy the orbitals (5 alpha electrons)
occ_model = AufbauOccModel(5)

# Converge WFN with CDIIS SCF
# - Construct the initial density matrix (needed for CDIIS).
occ_model.assign(exp_alpha)
dm_alpha = exp_alpha.to_dm()
# - SCF solver
scf_solver = CDIISSCFSolver(1e-6)
scf_solver(ham, lf, olp, occ_model, dm_alpha)

# Derive orbitals (coeffs, energies and occupations) from the Fock and density
# matrices. The energy is also computed to store it in the output file below.
fock_alpha = lf.create_two_index()
ham.reset(dm_alpha)
ham.compute_energy()
ham.compute_fock(fock_alpha)
exp_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)

# Assign results to the molecule object and write it to a file, e.g. for
# later analysis. Note that the CDIIS algorithm can only really construct an
# optimized density matrix and no orbitals.
mol.title = 'RKS computation on water'
mol.energy = ham.cache['energy']
mol.obasis = obasis
mol.exp_alpha = exp_alpha
mol.dm_alpha = dm_alpha

# useful for post-processing (results stored in double precision):
mol.to_file('water.h5')


# CODE BELOW IS FOR horton-regression-test.py ONLY. IT IS NOT PART OF THE EXAMPLE.
rt_results = {
    'energy': ham.cache['energy'],
    'exp_alpha': exp_alpha.energies,
    'nn': ham.cache["energy_nn"],
    'kin': ham.cache["energy_kin"],
    'ne': ham.cache["energy_ne"],
    'grid': ham.cache["energy_grid_group"],
    'hartree': ham.cache["energy_hartree"],
    'x_hf': ham.cache["energy_x_hf"],
}
# BEGIN AUTOGENERATED CODE. DO NOT CHANGE MANUALLY.
import numpy as np  # pylint: disable=wrong-import-position
rt_previous = {
    'energy': -76.372223106410885,
    'exp_alpha': np.array([
        -19.174675917533499, -1.0216889289766689, -0.54324149010045464,
        -0.37631403914157158, -0.30196183487620326, 0.079896573985756419,
        0.16296304612701332, 0.81419059490960388, 0.86377461055569127, 0.9243929453024935,
        0.95050094195149326, 1.1033737076332981, 1.4108569929549999, 1.7561523962868733,
        1.761532111350379, 1.8055689722633752, 2.3348442517458823, 2.6275437456471868
    ]),
    'grid': -6.821114560989138,
    'hartree': 46.93245844915478,
    'kin': 76.05549816546615,
    'ne': -199.18635862588496,
    'nn': 9.1571750364299866,
    'x_hf': -2.50988157058769,
}
