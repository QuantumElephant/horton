#!/usr/bin/env python
#JSON {"lot": "UKS/6-31G(d)",
#JSON  "scf": "CDIISSCFSolver",
#JSON  "er": "cholesky",
#JSON  "difficulty": 4,
#JSON  "description": "Basic UKS DFT example with GGA exhange-correlation functional (PBE)"}

import numpy as np
from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import


# Load the coordinates from file.
# Use the XYZ file from HORTON's test data directory.
fn_xyz = context.get_fn('test/methyl.xyz')
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
orb_beta = Orbitals(obasis.nbasis)

# Initial guess
guess_core_hamiltonian(olp, kin + na, orb_alpha, orb_beta)

# Construct the restricted HF effective Hamiltonian
external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
terms = [
    UTwoIndexTerm(kin, 'kin'),
    UDirectTerm(er_vecs, 'hartree'),
    UGridGroup(obasis, grid, [
        ULibXCGGA('x_pbe'),
        ULibXCGGA('c_pbe'),
    ]),
    UTwoIndexTerm(na, 'ne'),
]
ham = UEffHam(terms, external)

# Decide how to occupy the orbitals (5 alpha electrons, 4 beta electrons)
occ_model = AufbauOccModel(5, 4)

# Converge WFN with CDIIS SCF
# - Construct the initial density matrix (needed for CDIIS).
occ_model.assign(orb_alpha, orb_beta)
dm_alpha = orb_alpha.to_dm()
dm_beta = orb_beta.to_dm()
# - SCF solver
scf_solver = CDIISSCFSolver(1e-6)
scf_solver(ham, olp, occ_model, dm_alpha, dm_beta)

# Derive orbitals (coeffs, energies and occupations) from the Fock and density
# matrices. The energy is also computed to store it in the output file below.
fock_alpha = np.zeros(olp.shape)
fock_beta = np.zeros(olp.shape)
ham.reset(dm_alpha, dm_beta)
ham.compute_energy()
ham.compute_fock(fock_alpha, fock_beta)
orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
orb_beta.from_fock_and_dm(fock_beta, dm_beta, olp)

# Assign results to the molecule object and write it to a file, e.g. for
# later analysis. Note that the CDIIS algorithm can only really construct an
# optimized density matrix and no orbitals.
mol.title = 'UKS computation on methyl'
mol.energy = ham.cache['energy']
mol.obasis = obasis
mol.orb_alpha = orb_alpha
mol.orb_beta = orb_beta
mol.dm_alpha = dm_alpha
mol.dm_beta = dm_beta

# useful for post-processing (results stored in double precision):
mol.to_file('methyl.h5')

# CODE BELOW IS FOR horton-regression-test.py ONLY. IT IS NOT PART OF THE EXAMPLE.
rt_results = {
    'energy': ham.cache['energy'],
    'orb_alpha': orb_alpha.energies,
    'orb_beta': orb_beta.energies,
    'nn': ham.cache["energy_nn"],
    'kin': ham.cache["energy_kin"],
    'ne': ham.cache["energy_ne"],
    'grid': ham.cache["energy_grid_group"],
    'hartree': ham.cache["energy_hartree"],
}
# BEGIN AUTOGENERATED CODE. DO NOT CHANGE MANUALLY.
rt_previous = {
    'energy': -39.760858442576279,
    'orb_alpha': np.array([
        -9.9132239986929793, -0.59539168273588328, -0.35520825880899093,
        -0.35520428071873117, -0.18283296768425183, 0.065348689380477429,
        0.13500435708936726, 0.13501335248576246, 0.49484151010262456,
        0.53227791058244789, 0.53228034200278063, 0.64203475097075868,
        0.81497418080272854, 0.81499047699207794, 0.87633072588425232, 1.5835316698336181,
        1.5835584213834502, 1.8780344645378215, 2.0335501256098931, 2.0336331401576646
    ]),
    'orb_beta': np.array([
        -9.899085699262832, -0.55959622753340121, -0.34567489369825666,
        -0.34566542179299153, -0.078955929119382456, 0.083524761485062507,
        0.14222781839459014, 0.14224643591633562, 0.54494034344710285,
        0.54495006197900131, 0.57474884787842728, 0.66607960932832244,
        0.81847070490015816, 0.81850625147077816, 0.9130955405362291, 1.66323493666399,
        1.6634452544853966, 1.9879830719326739, 2.0463040592255393, 2.0463345823040369
    ]),
    'grid': -6.364465766881222,
    'hartree': 28.095039458858814,
    'kin': 39.32313924326898,
    'ne': -109.89435632048648,
    'nn': 9.0797849426636361,
}
