#!-*- coding: utf-8 -*-

import utils
import numpy as np
from time import perf_counter_ns
from settingsmodule import load_setting
from copy import deepcopy
from pyscf import gto, scf, dft

class pyscf_manager:
    
    def __init__(self, settings, atoms):

        self.gto = gto
        self.scf = scf
        self.dft = dft

        self.mol = gto.Mole()
        #self.mol.unit = 'Angstrom'
        self.mol.unit = 'Bohr'
        self.mol.atom = atoms
        self.mol.basis = load_setting(settings, 'ao_basis')
        self.mol.symmetry = False
        self.mol.verbose = 4
        self.mol.build()

        self.xc = load_setting(settings, 'xc')
        self.max_cycle = 500

        self.ks = self.dft.KS(self.mol) # currently only RKS is supprted (mol.spin==0 assumed)
        self.ks.xc = self.xc
        self.ks.max_cycle = self.max_cycle
        
        return


    def update_geometry(self, coords):
        
        atoms = deepcopy(self.mol._atom)

        elems = [ atom[0] for atom in atoms ]

        new_atoms = [ [ elem, coord ] for elem, coord in zip(elems, coords) ]

        self.mol.atom = new_atoms

        self.mol.build()

        self.ks = self.dft.KS(self.mol) # currently only RKS is supprted (mol.spin==0 assumed)
        self.ks.xc = self.xc
        self.ks.max_cycle = self.max_cycle

        return


    def converge_scf(self, dm = None):

        self.ks.kernel(dm = dm)

        return


    def return_hamiltonian(self, rho, n_spin):
        
        hcore = self.scf.hf.get_hcore(self.mol)
        ts = perf_counter_ns()
        veff  = self.ks.get_veff(dm = rho)
        te = perf_counter_ns()
        print("Time for veff: %12.3f sec.\n" % ((te-ts)/1.0e+9)) ## Debug code

        H = hcore + veff

        if n_spin == 1:

            return np.array( [ H ] )

        else:

            utils.stop_with_error('Currently not compatible with open-shell systems.')

    
    def get_overlap_matrix(self):
        
        return deepcopy( self.mol.intor('int1e_ovlp') )


    def get_derivative_coupling(self, velocity_2d):
        
        ipovlp = deepcopy( self.mol.intor('int1e_ipovlp') )
        # int1e_ipovlp is derivative w.r.t. electronic coordinate -> minus of nuclear derivative. But...
        # <dp/dr|q> = -<dp/dR|q> = <p|dq/dR> ?

        atom_indices = [ int(line.split()[0]) for line in self.mol.ao_labels() ]

        n_ao = len(atom_indices)

        deriv_coupling = np.zeros( (n_ao, n_ao), dtype='float64' )

        for i_ao in range(n_ao):

            i_atom = atom_indices[i_ao]
            
            vel = velocity_2d[i_atom]

            deriv_coupling[i_ao, :] = np.dot(vel, ipovlp[0:3, i_ao, :])
        
        #deriv_coupling = -deriv_coupling.transpose() # ?
        #deriv_coupling = deriv_coupling.transpose() # ?
        #deriv_coupling = -deriv_coupling # ?
        #deriv_coupling[:,:] = 0.0 ## Debug code

        return deepcopy(deriv_coupling)
