# -*- coding: utf-8 -*-
# Horton is a development platform for electronic structure methods.
# Copyright (C) 2011-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--
"""Two- and four-dimensional matrix implementations

   The purpose of this module is to provide a generic API for different
   implementations of real-valued double precision matrix storage and
   operations.

   Two-dimensional matrices are supposed to be symmetric and are used to
   represent one-body operators and 1DRDMs. Four-dimensional matrices are used
   to represent two-body operators, which are invariant under the following
   interchanges of indexes::

            <ij|kl> = <ji|lk> = <kl|ij> = <lk|ji> =
            <il|kj> = <jk|li> = <kj|il> = <li|jk>

   This module assumes physicists notation for the two-particle operators. It is
   up to the specific implementations of the matrices to make use of these
   symmetries.

   One should use these matrix implementations without accessing the internals
   of each class, i.e. without accessing attributes or methods that start with
   an underscore.

   In order to avoid temporaries when working with arrays, the methods do
   not return arrays. Instead such methods are an in place operation or have
   output arguments. This forces the user to allocate all memory in advance,
   which can then be moved out of the loops. The initial implementation (the
   Dense... classes) are just a proof of concept and may therefore contain
   internals that still make temporaries. This fixed later with an alternative
   implementation.
"""


import numpy as np
import mpmath as mp
import scipy as scp

from horton.log import log


__all__ = [
    'LinalgFactory', 'LinalgObject', 'Expansion', 'OneBody',
    'DenseLinalgFactory', 'DenseExpansion', 'DenseOneBody', 'DenseTwoBody',
]


class LinalgFactory(object):
    """A collection of compatible matrix and linear algebra routines.

       This is just an abstract base class that serves as a template for
       specific implementations.
    """
    def __init__(self, default_nbasis=None):
        '''
           **Optional arguments:**

           default_nbasis
                The default basis size when constructing new
                operators/expansions.
        '''
        self._default_nbasis = default_nbasis

    def set_default_nbasis(self, nbasis):
        self._default_nbasis = nbasis

    def create_expansion(self, nbasis=None):
        raise NotImplementedError

    def create_one_body(self, nbasis=None):
        raise NotImplementedError
    
    def create_one_body_from(self, A):
        raise NotImplementedError

    def create_two_body(self, nbasis=None):
        raise NotImplementedError

    def error_eigen(self, ham, overlap, expansion, epsilons):
        raise NotImplementedError

    def diagonalize(self, ham, overlap, expansion, epsilons):
        raise NotImplementedError

    def get_memory_one_body(self, nbasis=None):
        raise NotImplementedError

    def get_memory_two_body(self, nbasis=None):
        raise NotImplementedError


class LinalgObject(object):
    def apply_basis_permutation(self, permutation):
        raise NotImplementedError

    def apply_basis_signs(self, signs):
        raise NotImplementedError

    @classmethod
    def from_hdf5(cls, grp, lf):
        raise NotImplementedError

    def to_hdf5(self, grp):
        raise NotImplementedError

    def __clear__(self):
        self.clear()

    def clear(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def assign(self, other):
        raise NotImplementedError


class Expansion(LinalgObject):
    def __init__(self, nbasis, nfn=None):
        raise NotImplementedError

    def check_normalization(self, olp, eps=1e-4):
        raise NotImplementedError


class OneBody(LinalgObject):
    def __init__(self, nbasis):
        raise NotImplementedError

    def set_element(self, i, j, value):
        raise NotImplementedError

    def get_element(self, i, j):
        raise NotImplementedError

    def iadd(self, other, factor=1):
        raise NotImplementedError

    def expectation_value(self, dm):
        raise NotImplementedError

    def trace(self):
        raise NotImplementedError

    def itranspose(self):
        raise NotImplementedError

    def iscale(self, factor):
        raise NotImplementedError

    def dot(self, vec0, vec1):
        raise NotImplementedError

#     @classmethod
#     def matrix_product(cls, *args):
#         result = args[0].copy()
#         reduce(lambda x,y: x.imul(y), args[1:], result)
#         return result
    
    def imuls(self, *args):
        reduce(lambda x,y: x.imul(y), args, self)
    
#     @classmethod
#     def add_matrix(cls, *args):
#         result = args[0].copy()
#         reduce(lambda x,y: x.iadd(y), args[1:], result)
#         return result
        
    def iadds(self, *args):
        reduce(lambda x,y: x.iadd(y), args, self)
    
#     @classmethod
#     def sub_matrix(cls, *args):
#         result = args[0].copy()
#         reduce(lambda x,y: x.isub(y), args[1:], result)
#         return result
        
    def isubs(self, *args):
        reduce(lambda x,y: x.isub(y), args, self)

class DenseLinalgFactory(LinalgFactory):
    def create_expansion(self, nbasis=None, nfn=None):
        nbasis = nbasis or self._default_nbasis
        return DenseExpansion(nbasis, nfn)

    def _check_expansion_init_args(self, expansion, nbasis=None, nfn=None):
        nbasis = nbasis or self._default_nbasis
        expansion.__check_init_args__(nbasis, nfn)

    create_expansion.__check_init_args__ = _check_expansion_init_args


    def create_one_body(self, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        return DenseOneBody(nbasis)
    
    def create_one_body_from(self,A):
        result = DenseOneBody(A.shape[0])
        result._array = A
        return result
    
    def create_one_body_eye(self, nbasis=None):
        result = self.create_one_body_from(np.eye(nbasis or self._default_nbasis))
        return result

    def _check_one_body_init_args(self, one_body, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        one_body.__check_init_args__(nbasis)

    create_one_body.__check_init_args__ = _check_one_body_init_args
    create_one_body_from.__check_init_args__ = _check_one_body_init_args
    create_one_body_eye.__check_init_args__ = _check_one_body_init_args


    def create_two_body(self, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        return DenseTwoBody(nbasis)

    def _check_two_body_init_args(self, two_body, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        two_body.__check_init_args__(nbasis)

    create_two_body.__check_init_args__ = _check_two_body_init_args


    @staticmethod
    def error_eigen(fock, overlap, expansion):
        """Compute the error of the orbitals with respect to the eigenproblem

           **Arguments:**

           fock
                A DenseOneBody Hamiltonian (or Fock) operator.

           overlap
                A DenseOneBody overlap operator.

           expansion
                An expansion object containing the current orbitals/eginvectors.
        """
        errors = np.dot(fock._to_numpy(), expansion.coeffs) \
                 - expansion.energies*np.dot(overlap._to_numpy(), expansion.coeffs) #TODO: figure out how to remove direct np operations
        return np.sqrt((errors**2).mean())


    @staticmethod
    def diagonalize(fock, overlap=None):
        """Generalized eigen solver for the given Hamiltonian and overlap.

           **Arguments:**

           fock
                A DenseOneBody Hamiltonian (or Fock) operator.

           overlap
                A DenseOneBody overlap operator.

        """
        from scipy.linalg import eigh
        if overlap is None:
            return eigh(fock._to_numpy())
        else:
            return eigh(fock._to_numpy(), overlap._to_numpy())

    def get_memory_one_body(self, nbasis=None):
        return nbasis**2*8

    def get_memory_two_body(self, nbasis=None):
        return nbasis**4*8

    @classmethod
    def IVtoDense(cls, *vars):
        print "changing back to DenseOneBody, max uncertainty:"
        print [max([j.delta for j in i._array]) for i in vars]
#         assert False
        return [cls.create_one_body_from(i._to_numpy()) for i in vars]

class DenseExpansion(LinalgObject):
    """An expansion of several functions in a basis with a dense matrix of
       coefficients. The implementation is such that the columns of self._array
       contain the orbitals.
    """
    def __init__(self, nbasis, nfn=None):
        """
           **Arguments:**

           nbasis
                The number of basis functions.

           **Optional arguments:**

           nfn
                The number of functions to store. Defaults to nbasis.

           do_energies
                Also allocate an array to store an energy corresponding to each
                function.
        """
        if nfn is None:
            nfn = nbasis
        self._coeffs = np.zeros((nbasis, nfn), float)
        self._energies = np.zeros(nfn, float)
        self._occupations = np.zeros(nfn, float)
        log.mem.announce(self._coeffs.nbytes + self._energies.nbytes + self._occupations.nbytes)

    def __del__(self):
        if log is not None:
            log.mem.denounce(self._coeffs.nbytes + self._energies.nbytes + self._occupations.nbytes)

    def __check_init_args__(self, nbasis, nfn=None):
        if nfn is None:
            nfn = nbasis
        assert nbasis == self.nbasis
        assert nfn == self.nfn

    def read_from_hdf5(self, grp):
        if grp.attrs['class'] != self.__class__.__name__:
            raise TypeError('The class of the expansion in the HDF5 file does not match.')
        grp['coeffs'].read_direct(self._coeffs)
        grp['energies'].read_direct(self._energies)
        grp['occupations'].read_direct(self._occupations)

    def to_hdf5(self, grp):
        grp.attrs['class'] = self.__class__.__name__
        grp['coeffs'] = self._coeffs
        grp['energies'] = self._energies
        grp['occupations'] = self._occupations

    def _get_nbasis(self):
        '''The number of basis functions'''
        return self._coeffs.shape[0]

    nbasis = property(_get_nbasis)

    def _get_nfn(self):
        '''The number of orbitals (or functions in general)'''
        return self._coeffs.shape[1]

    nfn = property(_get_nfn)

    def _get_coeffs(self):
        '''The matrix with the expansion coefficients'''
        return self._coeffs.view()

    coeffs = property(_get_coeffs)

    def _get_energies(self):
        '''The orbital energies'''
        return self._energies

    energies = property(_get_energies)

    def _get_occupations(self):
        '''The orbital occupations'''
        return self._occupations

    occupations = property(_get_occupations)

    def clear(self):
        self._coeffs[:] = 0.0
        self._energies[:] = 0.0
        self._occupations[:] = 0.0

    def copy(self):
        result = DenseExpansion(self.nbasis, self.nfn)
        result._coeffs[:] = self._coeffs
        result._energies[:] = self._energies
        result._occupations[:] = self._occupations
        return result

    def check_normalization(self, olp, eps=1e-4):
        '''Run an internal test to see if the orbitals are normalized

           **Arguments:**

           olp
                The overlap one_body operators

           **Optional arguments:**

           eps
                The allowed deviation from unity, very loose by default.
        '''
        for i in xrange(self.nfn):
            if self.occupations[i] == 0:
                continue
            norm = olp.dot(self._coeffs[:,i], self._coeffs[:,i])
            print i, norm
            assert abs(norm-1) < eps, 'The orbitals are not normalized!'

    def compute_density_matrix(self, dm, factor=None):
        """Compute the density matrix

           **Arguments:**

           dm
                An output density matrix. This must be a DenseOneBody instance.

           **Optional arguments:**

           factor
                When given, the density matrix is added with the given prefactor
                to the output argument. If not given, the original contents of
                dm are overwritten.
        """
        if factor is None:
            dm.assign_from(np.dot(self._coeffs*self.occupations, self._coeffs.T))
        else:
            dm += factor*np.dot(self._coeffs*self.occupations, self._coeffs.T)

    def derive_from_fock_matrix(self, fock, overlap):
        '''Diagonalize a Fock matrix to obtain orbitals and energies'''
        evals, evecs = DenseLinalgFactory.diagonalize(fock, overlap)
        self._energies[:] = evals[:self.nfn]
        self._coeffs[:] = evecs[:,:self.nfn]

    def derive_from_density_and_fock_matrix(self, dm, fock, overlap, scale=-0.001):
        '''
           **Arguments**:

           dm
                A DenseOneBody object with the density matrix

           fock
                A DenseOneBody object with the Fock matrix

           overlap
                A DenseOneBody object with the overlap matrix

           **Optional arguments:**

           scale
                The linear coefficient for the density matrix. It is added to
                the Fock matrix as in level shifting to obtain a set of orbitals
                that diagonalizes both matrices.

           This only works well for slater determinants without (fractional)
           holes below the Fermi level.
        '''
        # Construct a level-shifted fock matrix to separate out the degenerate
        # orbitals with different occupations
        occ = overlap.copy()
        occ.idot(dm)
        occ.idot(overlap)
        tmp = fock.copy()
        tmp.iadd(occ, factor=scale)
        # diagonalize and compute eigenvalues
        evals, evecs = DenseLinalgFactory.diagonalize(tmp, overlap)
        self._coeffs[:] = evecs[:,:self.nfn]
        for i in xrange(self.nfn):
            orb = evecs[:,i]
            self._energies[i] = fock.dot(orb, orb)
            self._occupations[i] = occ.dot(orb, orb)

    def derive_naturals(self, dm, overlap):
        '''
           **Arguments**:

           dm
                A DenseOneBody object with the density matrix

           overlap
                A DenseOneBody object with the overlap matrix

           **Optional arguments:**
        '''
        # Construct a level-shifted operator
        occ = overlap.copy()
        occ.idot(dm)
        occ.idot(overlap)
        # diagonalize and compute eigenvalues
        evals, evecs = DenseLinalgFactory.diagonalize(occ, overlap)
        self._coeffs[:] = evecs[:,:self.nfn]
        self._occupations[:] = evals
        self._energies[:] = 0.0

    def apply_basis_permutation(self, permutation):
        '''Reorder the coefficients for a given permutation of basis functions.
        '''
        self._coeffs[:] = self.coeffs[permutation]

    def apply_basis_signs(self, signs):
        '''Correct for different sign conventions of the basis functions.'''
        self._coeffs *= signs.reshape(-1,1)

    def assign(self, other):
        if not isinstance(other, DenseExpansion):
            raise TypeError('The other object must also be DenseExpansion instance.')
        self._coeffs[:] = other._coeffs
        self._energies[:] = other._energies
        self._occupations[:] = other._occupations

    def get_homo_index(self, offset=0):
        '''Return the index of a HOMO orbital.'''
        if offset < 0:
            raise ValueError('Offset must be zero or positive.')
        homo_indexes = self.occupations.nonzero()[0]
        if len(homo_indexes) > offset:
            return homo_indexes[len(homo_indexes)-offset-1]

    def get_homo_energy(self, offset=0):
        '''Return a homo energy

           **Optional arguments**:

           offset
                By default, the (highest) homo energy is returned. When this
                index is above zero, the corresponding lower homo energy is
                returned.
        '''
        index = self.get_homo_index(offset)
        if index is not None:
            return self.energies[index]

    homo_energy = property(get_homo_energy)

    def get_lumo_index(self, offset=0):
        '''Return the index of a LUMO orbital.'''
        if offset < 0:
            raise ValueError('Offset must be zero or positive.')
        lumo_indexes = (self.occupations==0.0).nonzero()[0]
        if len(lumo_indexes) > offset:
            return lumo_indexes[offset]

    def get_lumo_energy(self, offset=0):
        '''Return a lumo energy

           **Optional arguments**:

           offset
                By default, the (lowest) lumo energy is returned. When this
                index is above zero, the corresponding higher homo energy is
                returned.
        '''
        index = self.get_lumo_index(offset)
        if index is not None:
            return self.energies[index]

    lumo_energy = property(get_lumo_energy)


class DenseOneBody(OneBody):
    """Dense symmetric two-dimensional matrix, also used for density matrices.

       This is the most inefficient implementation in terms of memory usage and
       computer time. Due to its simplicity, it is trivial to implement. This
       implementation mainly serves as a reference for testing purposes.
    """
    def __init__(self, nbasis):
        """
           **Arguments:**

           nbasis
                The number of basis functions.
        """
        self._array = np.zeros((nbasis, nbasis), float)
        log.mem.announce(self._array.nbytes)
        
        self._dense_array = self._array

    def __del__(self):
        if log is not None:
            log.mem.denounce(self._array.nbytes)

    def __check_init_args__(self, nbasis):
        assert nbasis == self.nbasis

    @classmethod
    def from_hdf5(cls, grp, lf):
        nbasis = grp['array'].shape[0]
        result = cls(nbasis)
        grp['array'].read_direct(result._array)
        return result

    def read_from_hdf5(self, grp):
        if grp.attrs['class'] != self.__class__.__name__:
            raise TypeError('The class of the one-body operator in the HDF5 file does not match.')
        grp['array'].read_direct(self._array)

    def to_hdf5(self, grp):
        grp.attrs['class'] = self.__class__.__name__
        grp['array'] = self._array

    def _get_nbasis(self):
        '''The number of basis functions'''
        return self._array.shape[0]

    nbasis = property(_get_nbasis)

    def set_element(self, i, j, value):
        self._array[i,j] = value
        self._array[j,i] = value

    def get_element(self, i, j):
        return self._array[i,j]

    def assign(self, other):
        if not isinstance(other, DenseOneBody):
            try:
                other = other.to_dense_one_body()
            except AttributeError:
                assert False,("the other object must have a toDenseOneBody method",type(other))
        self._array[:] = other._array

    def copy(self):
        result = DenseOneBody(self.nbasis)
        result._array[:] = self._array
        return result

    def check_symmetry(self):
        '''Check the symmetry of the array. For testing only.'''
        assert abs(self._array - self._array.T).max() == 0.0

    def clear(self):
        self._array[:] = 0.0

    def iadd(self, other, factor=1):
        self._array += other._array*factor
        return self
    
    def isub(self, other):
        self.iadd(other, -1)
        return self

    def imul(self, other):
        if isinstance(other, DenseOneBody): #TODO: Abstract out dense requirement
            self._array = np.dot(self._array, other._array)
        else:
            assert isinstance(other, int) or isinstance(other, float)
            self.iscale(other)
        return self

    def expectation_value(self, dm):
        return np.dot(self._array.ravel(), dm._array.ravel())

    def trace(self):
        return np.trace(self._array)

    def itranspose(self):
        '''In-place transpose'''
        self._array = self._array.T

    def iscale(self, factor):
        self._array *= factor

    def dot(self, vec0, vec1):
        return np.dot(vec0, np.dot(self._array, vec1))

    def idot(self, other):
        self._array[:] = np.dot(self._array, other._array)

    def distance(self, other):
        return abs(self._array.ravel() - other._array.ravel()).max()

    def apply_basis_permutation(self, permutation):
        '''Reorder the coefficients for a given permutation of basis functions.
        '''
        self._array[:] = self._array[permutation]
        self._array[:] = self._array[:,permutation]

    def apply_basis_signs(self, signs):
        '''Correct for different sign conventions of the basis functions.'''
        self._array *= signs
        self._array *= signs.reshape(-1,1)
    
    def isymmetrize(self):
        self._array = 0.5*(self._array + self._array.T)    
    
    def assign_from(self, A):
        assert isinstance(A, np.ndarray)
        self._array = A    
    
    def ravel(self):
        return self._array.ravel()
    
    def _to_numpy(self):
        return self._array
    
    def _to_dense_one_body(self):
        return self
    
    def _update_dense(self):
        self._dense_array = self._array.copy()
    def _update_array(self):
        self._array = self._dense_array.copy()
    
    def set_elements_from_vec(self, x):
        assert x.size == self.nbasis**2.
        self._array = x.reshape([self.nbasis,self.nbasis])
        
    def iscale_diag(self, factor):
        ind = np.diag_indices_from(self._array)
        self._array[ind] *= factor
        
    def invert(self):
        self._array = np.linalg.inv(self._array)
        
    def outer(self, eigvecs, eigvals=None):
        """
            **Arguments:**
            
            eigvecs
                An onebody instance of eigenvectors corresponding to non-zero eigenvalues.
                
            **Optional Arguments:**
            
            eigvals
                A 1d numpy array of eigenvalues. Zero eigenvalues will be skipped.
        """
        
        if eigvals is None:
            eigvals = np.ones(self._array.shape[0])
                
        for key,i in enumerate(eigvals[eigvals != 0]):
            if isinstance(eigvecs, DenseExpansion):
                internal_rep = eigvecs._coeffs
            else:
                internal_rep = eigvecs._array
                
            self._array += i*np.outer(internal_rep[:,key], internal_rep[:,key])
            
    
    def assign_from_blocks(self, *args):
        result = []
        for i in args:
            if isinstance(i, DenseExpansion): #update to Expansion
                result.append(i._coeffs)
            elif isinstance(i, OneBody):
                result.append(i._array)
            else:
                raise NotImplementedError
        
        self._array = scp.linalg.block_diag(*result)           
    
class DenseTwoBody(LinalgObject):
    """Dense symmetric four-dimensional matrix.

       This is the most inefficient implementation in terms of memory usage and
       computer time. Due to its simplicity, it is trivial to implement. This
       implementation mainly serves as a reference for testing purposes.
    """
    def __init__(self, nbasis):
        """
           **Arguments:**

           nbasis
                The number of basis functions.
        """
        self._array = np.zeros((nbasis, nbasis, nbasis, nbasis), float)
        log.mem.announce(self._array.nbytes)

    def __del__(self):
        if log is not None:
            log.mem.denounce(self._array.nbytes)

    def __check_init_args__(self, nbasis):
        assert nbasis == self.nbasis

    @classmethod
    def from_hdf5(cls, grp, lf):
        nbasis = grp['array'].shape[0]
        result = cls(nbasis)
        grp['array'].read_direct(result._array)
        return result

    def to_hdf5(self, grp):
        grp.attrs['class'] = self.__class__.__name__
        grp['array'] = self._array

    def _get_nbasis(self):
        '''The number of basis functions'''
        return self._array.shape[0]

    nbasis = property(_get_nbasis)

    def set_element(self, i, j, k, l, value):
        #    <ij|kl> = <ji|lk> = <kl|ij> = <lk|ji> =
        #    <il|kj> = <jk|li> = <kj|il> = <li|jk>
        self._array[i,j,k,l] = value
        self._array[j,i,l,k] = value
        self._array[k,l,i,j] = value
        self._array[l,k,j,i] = value
        self._array[i,l,k,j] = value
        self._array[j,k,l,i] = value
        self._array[k,j,i,l] = value
        self._array[l,i,j,k] = value

    def get_element(self, i, j, k, l):
        return self._array[i,j, k, l]

    def check_symmetry(self):
        """Check the symmetry of the array."""
        assert abs(self._array - self._array.transpose(1,0,3,2)).max() == 0.0
        assert abs(self._array - self._array.transpose(2,3,0,1)).max() == 0.0
        assert abs(self._array - self._array.transpose(3,2,1,0)).max() == 0.0
        assert abs(self._array - self._array.transpose(2,1,0,3)).max() == 0.0
        assert abs(self._array - self._array.transpose(3,0,1,2)).max() == 0.0
        assert abs(self._array - self._array.transpose(0,3,2,1)).max() == 0.0
        assert abs(self._array - self._array.transpose(1,2,3,0)).max() == 0.0

    def apply_direct(self, dm, output):
        """Compute the direct dot product with a density matrix."""
        if not isinstance(dm, OneBody):
            raise TypeError('The dm argument must be a DenseOneBody class')
        if not isinstance(output, OneBody):
            raise TypeError('The output argument must be a DenseOneBody class')
        output.assign_from(np.tensordot(self._array, dm._to_numpy(), ([1,3], [0,1])))

    def apply_exchange(self, dm, output):
        """Compute the exchange dot product with a density matrix."""
        if not isinstance(dm, OneBody):
            raise TypeError('The dm argument must be a DenseOneBody class')
        if not isinstance(output, OneBody):
            raise TypeError('The output argument must be a DenseOneBody class')
        output.assign_from(np.tensordot(self._array, dm._to_numpy(), ([1,2], [0,1])))

    def clear(self):
        self._array[:] = 0.0

    def apply_basis_permutation(self, permutation):
        '''Reorder the coefficients for a given permutation of basis functions.
        '''
        self._array[:] = self._array[permutation]
        self._array[:] = self._array[:,permutation]
        self._array[:] = self._array[:,:,permutation]
        self._array[:] = self._array[:,:,:,permutation]

    def apply_basis_signs(self, signs):
        '''Correct for different sign conventions of the basis functions.'''
        self._array *= signs
        self._array *= signs.reshape(-1,1)
        self._array *= signs.reshape(-1,-1,1)
        self._array *= signs.reshape(-1,-1,-1,1)

class TriangularLinalgFactory(DenseLinalgFactory):
    def _check_one_body_init_args(self, one_body, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        one_body.__check_init_args__(nbasis)
    
    def create_one_body(self, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        return TriangularOneBody(nbasis)
    
    create_one_body.__check_init_args__ = _check_one_body_init_args
    
    def create_one_body_from(self, A):
#         assert (np.abs(A-A.T) < 1e-10).all(), (A-A.T) #is it symmetric?
        result = TriangularOneBody(A.shape[0], A)
#         result.isymmetrize()
        return result
    
    create_one_body_from.__check_init_args__ = _check_one_body_init_args

    def get_memory_one_body(self, nbasis=None):
        return nbasis**2*8 #not technically correct. Still needs size of indices and indptr arrays

    def get_memory_two_body(self, nbasis=None):
        return nbasis**4*8

class TriangularOneBody(DenseOneBody):
    """Sparse upper triangular symmetric two-dimensional matrix, also used for density matrices.

        Internally stored as a scipy.sparse.csr_matrix (optimized for row operations).
    """
    def __init__(self, nbasis, A=None):
        """
           **Arguments:**

           nbasis
                The number of basis functions.
        """
        if A is None:
            self._array = np.zeros((nbasis, nbasis))
        else:
            self._array = A
        log.mem.announce(self._array.nbytes)

        self._dense_array = self._array
        
    def ravel(self):
        ind = np.triu_indices_from(self._array)
        result = self._array[ind].squeeze()
        return result 
    
    def copy(self):
        result = TriangularOneBody(self.nbasis)
        result._array[:] = self._array
        return result
    
    def set_elements_from_vec(self, x):
        assert x.size == self.nbasis*(self.nbasis+1)/2.
        ind = np.triu_indices_from(self._array)
        self._array[ind] = x
        self._array += np.triu(self._array, 1).T
        
    def _to_dense_one_body(self):
        result = DenseOneBody(self.nbasis)
        result.assign_from(self._array)
        return result

    def _update_dense(self):
        self._dense_array = self._to_numpy()
    def _update_array(self):
        self.assign_from(self._dense_array)

class BaseDualOneBody(TriangularOneBody):
    """Dense symmetric two-dimensional matrix, also used for density matrices.

       This is the most inefficient implementation in terms of memory usage and
       computer time. Due to its simplicity, it is trivial to implement. This
       implementation mainly serves as a reference for testing purposes.
    """
    def __init__(self, nbasis, A=None, isDual=False):
        """
           **Arguments:**

           nbasis
                The number of basis functions.
        """
        self.isDual = isDual
        
        super(BaseDualOneBody, self).__init__(nbasis)
        
        if A is not None:
            assert isinstance(A, np.ndarray)
            self._array = A

        if isDual:
            self._dense_to_dual()
        
        
    def enable_dual(self):
        self.isDual = True
        self._dense_to_dual()
        
    def _dense_to_dual(self):
        raise NotImplementedError
    
    def disable_dual(self, writeback=False):
        raise NotImplementedError
    
    def set_elements_from_vec(self, np_x):
        if self.isDual:
            assert np_x.size == self.nbasis*(self.nbasis+1)/2.
            ind = np.triu_indices(self.nbasis)
            for i in np.arange(np_x.size):
                self._dual_array[ind[0][i], ind[1][i]] = np_x[i]
                self._dual_array[ind[1][i], ind[0][i]] = np_x[i] #symmetry
            
        super(BaseDualOneBody, self).set_elements_from_vec(np_x)

    def set_element(self, i, j, value):
        if self.isDual:
            raise NotImplementedError
    #         self._dual_array[i,j] = value
    #         self._dual_array[j,i] = value
            
        super(BaseDualOneBody, self).set_element(i,j,value)

    def assign(self, other):
        if self.isDual:
            if not isinstance(other, BaseDualOneBody):
                assert False,"the other object must be a dual matrix"
            for i in np.arange(other._dual_array.rows):
                for j in np.arange(other._dual_array.cols):
                    self._dual_array[i,j] = other._dual_array[i,j]
        super(BaseDualOneBody, self).assign(other)

    def copy(self):
        raise NotImplementedError

    def clear(self):
        if self.isDual:
            for i in np.arange(self._dual_array.rows):
                for j in np.arange(self._dual_array.cols):
                    self._dual_array[i,j] = 0
        super(BaseDualOneBody, self).clear()

    def iadd(self, other, factor=1):
        if self.isDual:
            if not other.isDual:
                other.enable_dual()
            self._dual_array += other._dual_array*factor
        super(BaseDualOneBody, self).iadd(other, factor)
        return self
    
    def imul(self, other):
        if self.isDual:
            if isinstance(other, BaseDualOneBody): #TODO: Abstract out dense requirement
                if not other.isDual:
                    other.enable_dual()
                
                self._dual_array = self._dual_array*other._dual_array
            else:
                assert isinstance(other, int) or isinstance(other, float)
                self.iscale(other)
        super(BaseDualOneBody, self).imul(other)
        return self

    def expectation_value(self, dm):
        if self.isDual:
            result = 0
            temp = self._dual_array*dm._dual_array
            for i in np.arange(temp.rows):
                result += temp[i,i]
#             print "expectation error " + str(super(BaseDualOneBody, self).expectation_value(dm))
        else:
            result = super(BaseDualOneBody, self).expectation_value(dm)
        return result

    def trace(self):
        result = super(BaseDualOneBody, self).trace()
        if self.isDual:
            dual_result = 0
            for i in np.arange(self._dual_array.rows):
                dual_result += self._dual_array[i,i]
#             print "trace error: " + str(super(BaseDualOneBody, self).trace() - result)
#                 print "trace error: " + str(dual_result.delta)
        return result

    def itranspose(self):
        '''In-place transpose'''
        if self.isDual:
            self._dual_array = self._dual_array.T
        super(BaseDualOneBody, self).itranspose()

    def iscale(self, factor):
        if self.isDual:
            self._dual_array *= factor
        super(BaseDualOneBody, self).iscale(factor)

    def dot(self, vec0, vec1):
        result = super(BaseDualOneBody, self).dot(vec0, vec1)
#         if self.isDual:
#             
#             
#             result = vec0*self._dual_array*vec1 #FIXME: Multiply with numpy vectors
#             print "floating point dot error: " + str(super(BaseDualOneBody, self).dot(vec0, vec1) - result)
        return result

    def idot(self, other):
        if self.isDual:
            self._dual_array = self._dual_array*other._dual_array
        super(BaseDualOneBody, self).idot(other)

    def distance(self, other):
        result = super(BaseDualOneBody, self).distance(other)
        if self.isDual:
            result = max(self._dual_array - other._dual_array)
#             print "floating point distance error: " + str(super(BaseDualOneBody, self).distance(other) - result)
        return result

#     def apply_basis_permutation(self, permutation):
#         '''Reorder the coefficients for a given permutation of basis functions.
#         '''
#         self._array[:] = self._array[permutation]
#         self._array[:] = self._array[:,permutation]
# 
#     def apply_basis_signs(self, signs):
#         '''Correct for different sign conventions of the basis functions.'''
#         self._coeffs *= signs
#         self._coeffs *= signs.reshape(-1,1)
    
    def isymmetrize(self):
        if self.isDual:
            self._dual_array = 0.5*(self._dual_array + self._dual_array.T)
        super(BaseDualOneBody, self).isymmetrize()    
     
    def assign_from(self, A):
        assert isinstance(A, np.ndarray), type(A)
        super(BaseDualOneBody, self).assign_from(A)
        if self.isDual:
            self._dense_to_dual()
            
    def _update_array(self):
        super(BaseDualOneBody, self)._update_array()
        if self.isDual:
            self._dense_to_dual()
            
#     def _update_dense(self):
#         self._dense_array = self._array.copy()
#         super(BaseDualOneBody, self)._update_dense()
    
    def _writeback(self):
        raise NotImplementedError
    
    def _calc_dual_error(self):
        raise NotImplementedError
    
    def ravel(self):
        self._calc_dual_error()
        self._writeback()
        result = super(BaseDualOneBody, self).ravel()
        return result 
        
    def iscale_diag(self, factor):
        if self.isDual:
            for i in np.arange(self._dual_array.rows):
                self._dual_array[i,i] *= factor
        super(BaseDualOneBody, self).iscale_diag(factor)
        
    def invert(self):
#         if pseudo: #not implemented in MPMath
#             self._dual_array = mp.matrices.linalg.inverse(self._dual_array)
            
        super(BaseDualOneBody, self).invert()
        
    def outer(self, eigvecs, eigvals=None):
        """
            **Arguments:**
            
            eigvecs
                An onebody instance of eigenvectors corresponding to non-zero eigenvalues.
                
            **Optional Arguments:**
            
            eigvals
                A 1d array of eigenvalues. Zero eigenvalues will be skipped.
        """

        super(BaseDualOneBody, self).outer(eigvecs, eigvals)
        
        if self.isDual:
            if isinstance(eigvecs, DenseExpansion):
                internal_rep = self._convert_to_dual(eigvecs._coeffs)
            elif isinstance(eigvecs, np.ndarray):
                internal_rep = self._convert_to_dual(eigvecs._array)
            else:
                internal_rep = eigvecs
                
            for key,i in enumerate(eigvals[eigvals != 0]):
                self._dual_array += internal_rep[:,key]*internal_rep[:,key].T*i #no idea why, but scalar must be at the end
    
    def _convert_to_dual(self, arg):
        raise NotImplementedError
    
    def assign_from_blocks(self, *args):
        last_diag = 0
        for i in args:
            for j in np.arange(i.rows):
                for k in np.arange(i.cols):
                    self._dual_array[j+last_diag,k+last_diag] = i[j,k]
            last_diag += i.rows #or i.cols
        
        super(BaseDualOneBody, self).assign_from_blocks(*args)
    
class IVDualOneBody(BaseDualOneBody):
    def _dense_to_dual(self):
        self._dual_array = mp.iv.matrix(self._array)
    
    def _convert_to_dual(self, arg): #internal helper function. Do not use
        return mp.iv.matrix(arg)
    
    def disable_dual(self, writeback=False):
        if self.isDual:
            deltas = self._calc_dual_error()
            
            self.isDual = False
            if deltas > 1e-30:
                return deltas
    
    def _writeback(self):
        pass
    
    def _calc_dual_error(self):
        if self.isDual:
    #         mids = [i.mid for i in self._dual_array]
            deltas = [i.delta for i in self._dual_array]
            
#             print "max:" + str(max(deltas))
            
            if mp.norm(deltas) > 1e-5:
                raise AbnormalOperationException
            elif mp.norm(deltas) > 1e-10:
                print "large numerical error propagated: " + str(deltas.norm())
            
            return mp.norm(deltas)
    
    def copy(self):
        result = IVDualOneBody(self.nbasis, isDual=self.isDual)
        result.assign(self)
        return result
    
class MPDualOneBody(BaseDualOneBody):
    def _dense_to_dual(self):
        self._dual_array = mp.mp.matrix(self._array)
    
    def _convert_to_dual(self, arg): #internal helper function. Do not use
        return mp.mp.matrix(arg)
    
    def _writeback(self):
        if self.isDual:
            for i in np.arange(self._dual_array.rows):
                for j in np.arange(self._dual_array.cols):
                    self._array[i,j] = float(self._dual_array[i,j])
        else:
            print "Warning: arbitrary precision values not written back!" 
    
    def _calc_dual_error(self):
        if self.isDual:
            deltas = mp.matrix(self._dual_array.rows, self._dual_array.cols)
            for i in np.arange(self._dual_array.rows):
                for j in np.arange(self._dual_array.cols):
                    deltas[i,j] = abs(self._array[i,j] - self._dual_array[i,j])
    
            if mp.norm(deltas) > 1e-5:
                raise AbnormalOperationException
            elif mp.norm(deltas) > 1e-10:
                print "large numerical error present: " + str(deltas.norm())
            
            return mp.norm(deltas)
    
    def disable_dual(self, writeback=False):
        if self.isDual:
            deltas = self._calc_dual_error()
            if writeback:
                self._writeback()
            
            self.isDual = False
            if deltas > 1e-30:
                return deltas

    def copy(self):
        result = MPDualOneBody(self.nbasis, isDual=self.isDual)
        result.assign(self)
        return result
    
class BaseDualLinalgFactory(DenseLinalgFactory):
    def __init__(self, default_nbasis=None, isDual=False):
        self.isDual = isDual
        self.history = [] #Memory usage is going to spike. But can't get around without delete operation.
    
    def _check_one_body_init_args(self, one_body, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        one_body.__check_init_args__(nbasis)
    
    def enable_dual(self):
        self.isDual = True
        for i in self.history:
            i.enable_dual()
        
    def disable_dual(self):
        raise NotImplementedError
    
    def _instantiate_one_body(self, nbasis=None):
        raise NotImplementedError
    
    def create_one_body(self, nbasis=None):
        result = self._instantiate_one_body(nbasis)
        self.history.append(result)
        return result
    
    create_one_body.__check_init_args__ = _check_one_body_init_args
    
    def create_one_body_from(self,A):
        result = self.create_one_body(A.shape[0])
        result.isDual = self.isDual
        result.assign_from(A)
        return result
    
    create_one_body_from.__check_init_args__ = _check_one_body_init_args
    
    @classmethod
    def DensetoIV(cls, *vars):
        return [cls.create_one_body_from(i._array) for i in vars]
                
    
    
class IVDualLinalgFactory(BaseDualLinalgFactory):
    def _instantiate_one_body(self, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        result = IVDualOneBody(nbasis, isDual=self.isDual)
        return result
    
    def disable_dual(self):
        self.isDual = False
        for i in self.history:
            numError = i.disable_dual(writeback=False)
            
            if numError is not None:
                    print "On matrix " + hex(id(i)) + " in history, norm interval error: " + str(numError)
        self.history = []
class MPDualLinalgFactory(BaseDualLinalgFactory):
    def __init__(self, default_nbasis=None, isDual=False):
        mp.dps = 30
        super(MPDualLinalgFactory, self).__init__(default_nbasis, isDual)
    
    def _instantiate_one_body(self, nbasis=None):
        nbasis = nbasis or self._default_nbasis
        result = MPDualOneBody(nbasis, isDual=self.isDual)
        return result 
    
    def disable_dual(self):
        self.isDual = False
        for i in self.history:
            numError = i.disable_dual(writeback=True)
            
            if numError is not None:
                    print "On matrix " + hex(id(i)) + " in history, norm absolute error: " + str(numError)
        self.history = []
