#!/usr/bin/env python
#JSON {"lot": "RHF/6-31G",
#JSON  "scf": "PlainSCFSolver",
#JSON  "er": "cholesky",
#JSON  "difficulty": 1,
#JSON  "description": "Basic RHF example with Cholesky matrices, includes export of Hamiltonian"}

import numpy as np

from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import


# Hartree-Fock calculation
# ------------------------

# Construct a molecule from scratch
mol = IOData.from_file(context.get_fn('test/h2.xyz'))

# Create a Gaussian basis set
obasis = get_gobasis(mol.coordinates, mol.numbers, '6-31G')

# Compute Gaussian integrals
olp = obasis.compute_overlap()
kin = obasis.compute_kinetic()
na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
er_vecs = obasis.compute_electron_repulsion_cholesky()

# Create alpha orbitals
orb_alpha = Orbitals(obasis.nbasis)

# Initial guess
one = kin + na
guess_core_hamiltonian(olp, one, orb_alpha)

# Construct the restricted HF effective Hamiltonian
external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
terms = [
    RTwoIndexTerm(kin, 'kin'),
    RDirectTerm(er_vecs, 'hartree'),
    RExchangeTerm(er_vecs, 'x_hf'),
    RTwoIndexTerm(na, 'ne'),
]
ham = REffHam(terms, external)

# Decide how to occupy the orbitals (1 alpha electron)
occ_model = AufbauOccModel(1)

# Converge WFN with plain SCF
scf_solver = PlainSCFSolver(1e-6)
scf_solver(ham, olp, occ_model, orb_alpha)


# Write SCF results to a file
# ---------------------------

# Assign results to the molecule object and write it to a file, e.g. for
# later analysis
mol.title = 'RHF computation on dinitrogen'
mol.energy = ham.cache['energy']
mol.obasis = obasis
mol.orb_alpha = orb_alpha

# useful for visualization:
mol.to_file('h2-scf.molden')
# useful for post-processing (results stored in double precision)
mol.to_file('h2-scf.h5')


# Export Hamiltonian in Hartree-Fock molecular orbital basis (all orbitals active)
# --------------------------------------------------------------------------------

# Transform orbitals
(one_mo,), (two_mo,) = transform_integrals_cholesky(one, er_vecs, 'tensordot', mol.orb_alpha)

# Prepare an IOData object for writing the Hamiltonian.
mol_all_active = IOData(core_energy=external['nn'], one_mo=one_mo, two_mo=two_mo)
# The Cholesky decomposition can only be stored in the internal format.
mol_all_active.to_file('h2-hamiltonian.h5')


# CODE BELOW IS FOR horton-regression-test.py ONLY. IT IS NOT PART OF THE EXAMPLE.
rt_results = {
    'energy': ham.cache['energy'],
    'orb_alpha': orb_alpha.energies,
    'nn': ham.cache["energy_nn"],
    'kin': ham.cache["energy_kin"],
    'ne': ham.cache["energy_ne"],
    'hartree': ham.cache["energy_hartree"],
    'x_hf': ham.cache["energy_x_hf"],
}
# BEGIN AUTOGENERATED CODE. DO NOT CHANGE MANUALLY.
rt_previous = {
    'energy': -1.1267239967341496,
    'orb_alpha': np.array([
        -0.59521025080650336, 0.23793808577403591, 0.77568902766127312, 1.4017781983799871
    ]),
    'hartree': 1.2989586824630228,
    'kin': 1.1232515338620017,
    'ne': -3.6126317024206545,
    'nn': 0.71317683059299186,
    'x_hf': -0.6494793412315112,
}
