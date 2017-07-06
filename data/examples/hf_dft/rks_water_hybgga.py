#!/usr/bin/env python
#JSON {"lot": "RKS/6-31G(d)",
#JSON  "scf": "EDIIS2SCFSolver",
#JSON  "er": "cholesky",
#JSON  "difficulty": 5,
#JSON  "description": "Basic RKS DFT example with hyrbid GGA exhange-correlation functional (B3LYP)"}

import numpy as np
from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import


# Load the coordinates from file.
# Use the XYZ file from HORTON's test data directory.
fn_xyz = context.get_fn('test/water.xyz')
mol = IOData.from_file(fn_xyz)

# Create a Gaussian basis set
obasis = get_gobasis(mol.coordinates, mol.numbers, '6-31g(d)')

# Compute Gaussian integrals
olp = obasis.compute_overlap()
kin = obasis.compute_kinetic()
na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
er_vecs = obasis.compute_electron_repulsion_cholesky()

# Define a numerical integration grid needed the XC functionals
grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers)

# Create alpha orbitals
orb_alpha = Orbitals(obasis.nbasis)

# Initial guess
guess_core_hamiltonian(olp, kin + na, orb_alpha)

# Construct the restricted HF effective Hamiltonian
external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
libxc_term = RLibXCHybridGGA('xc_b3lyp')
terms = [
    RTwoIndexTerm(kin, 'kin'),
    RDirectTerm(er_vecs, 'hartree'),
    RGridGroup(obasis, grid, [libxc_term]),
    RExchangeTerm(er_vecs, 'x_hf', libxc_term.get_exx_fraction()),
    RTwoIndexTerm(na, 'ne'),
]
ham = REffHam(terms, external)

# Decide how to occupy the orbitals (5 alpha electrons)
occ_model = AufbauOccModel(5)

# Converge WFN with CDIIS+EDIIS SCF
# - Construct the initial density matrix (needed for CDIIS+EDIIS).
occ_model.assign(orb_alpha)
dm_alpha = orb_alpha.to_dm()
# - SCF solver
scf_solver = EDIIS2SCFSolver(1e-6)
scf_solver(ham, olp, occ_model, dm_alpha)

# Derive orbitals (coeffs, energies and occupations) from the Fock and density
# matrices. The energy is also computed to store it in the output file below.
fock_alpha = np.zeros(olp.shape)
ham.reset(dm_alpha)
ham.compute_energy()
ham.compute_fock(fock_alpha)
orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)

# Assign results to the molecule object and write it to a file, e.g. for
# later analysis. Note that the CDIIS_EDIIS algorithm can only really construct
# an optimized density matrix and no orbitals.
mol.title = 'RKS computation on water'
mol.energy = ham.cache['energy']
mol.obasis = obasis
mol.orb_alpha = orb_alpha
mol.dm_alpha = dm_alpha

# useful for post-processing (results stored in double precision):
mol.to_file('water.h5')


# CODE BELOW IS FOR horton-regression-test.py ONLY. IT IS NOT PART OF THE EXAMPLE.
rt_results = {
    'energy': ham.cache['energy'],
    'orb_alpha': orb_alpha.energies,
    'nn': ham.cache["energy_nn"],
    'kin': ham.cache["energy_kin"],
    'ne': ham.cache["energy_ne"],
    'grid': ham.cache["energy_grid_group"],
    'hartree': ham.cache["energy_hartree"],
    'x_hf': ham.cache["energy_x_hf"],
}
# BEGIN AUTOGENERATED CODE. DO NOT CHANGE MANUALLY.
rt_previous = {
    'energy': -76.406156776346975,
    'orb_alpha': np.array([
        -19.12494652215198, -0.99562109649344044, -0.52934359625260619,
        -0.35973919172781244, -0.28895110439599314, 0.068187099284877942,
        0.1532902668612677, 0.80078130036326101, 0.84958389626115138, 0.89305132504935913,
        0.92182191946355896, 1.074508959522454, 1.3767806620540104, 1.7405943781554678,
        1.7462666980125516, 1.7861275433424106, 2.3057917944397714, 2.5943014303914662
    ]),
    'grid': -7.568923843396495,
    'hartree': 46.893530019953076,
    'kin': 76.03393036526309,
    'ne': -199.129803256826,
    'nn': 9.1571750364299866,
    'x_hf': -1.792065097770653,
}
