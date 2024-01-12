#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy.linalg as sp
from copy import deepcopy

from constants import H_DIRAC, ANGST2AU
import utils

from interface_dftbplus import dftbplus_manager

class Electronic_state:
    

    def __init__(self, settings, atomparams, e_coeffs, position, velocity, t, construct_initial_gs = True):

        self.active_occ_mos      = settings['active_occ_mos']
        self.active_vir_mos      = settings['active_vir_mos']
        self.qc_program          = settings['engine']['type']
        
        self.e_coeffs            = e_coeffs # np.array (n_estate)
        self.old_e_coeffs        = deepcopy(self.e_coeffs)
        self.estate_energies     = None
        self.old_estate_energies = None
        self.old_position        = deepcopy(position)
        self.old_velocity        = deepcopy(velocity)
        self.position            = deepcopy(position)
        self.velocity            = deepcopy(velocity)
        self.H                   = None
        self.S                   = None
        self.atomparams          = deepcopy(atomparams)

        self.is_open_shell       = False

        self.e_coeffs_tderiv     = np.zeros_like(e_coeffs)
        self.old_e_coeffs_tderiv = np.zeros_like(e_coeffs)
        
        self.t_e_coeffs           = t
        self.t_molecular_orbitals = sys.maxsize
        self.t_estate_energies    = sys.maxsize
        self.t_matrices           = sys.maxsize
        self.t_force              = sys.maxsize
        self.t_tdnac              = sys.maxsize
        self.t_state_coeffs       = sys.maxsize

        if construct_initial_gs:
            self.construct_initial_gs()
        else:
            self.gs_energy   = None
            self.gs_filling  = None
            self.mo_energies = None
            self.mo_coeffs   = None
            self.n_elec      = None
            self.n_MO        = None
            self.n_AO        = None
            self.gs_rho      = None

        return


    def set_new_position(self, position):
        self.old_position = deepcopy(self.position)
        self.position = deepcopy(position)
        return

    def set_new_velocity(self, velocity):
        self.old_velocity = deepcopy(self.velocity)
        self.velocity = deepcopy(velocity)
        return

    def set_new_time(self, t):
        self.t = t
        return


    def is_uptodate(self, time_last):
        
        if utils.is_equal_scalar(self.get_t(), time_last):
            return True
        else:
            return False


    def get_t(self):
        return self.t


    def get_n_estate(self):
        return len(self.e_coeffs)


    def get_n_occ(self):
        return self.n_occ


    def get_gs_energy(self):
        return self.gs_energy


    def get_atomparams(self):
        return deepcopy(self.atomparams)


    def get_H(self):

        if not self.is_uptodate(self.t_matrices):
            self.update_matrices()

        return deepcopy(self.H)


    def get_S(self):

        if not self.is_uptodate(self.t_matrices):
            self.update_matrices()

        return deepcopy(self.S)


    def get_molecular_orbitals(self):
        """Solve F * C = S * C * e ; get MO energies and coefficients."""    

        self.update_molecular_orbitals()

        return deepcopy(self.mo_energies), deepcopy(self.mo_coeffs)


    def construct_initial_molecular_orbitals(self):
        """Solve F * C = S * C * e ; get MO energies and coefficients."""    
        H = self.get_H()
        S = self.get_S()

        self.mo_energies, self.mo_coeffs = sp.eigh(H, S, type=1)
        #self.mo_energies, self.mo_coeffs = np.linalg.eigh(H, S)

        self.t_mos = self.get_t()

        return


    def update_molecular_orbitals(self):
        """Update MO energies and coefficients according to TD-KS equation."""
        utils.printer.write_out('Updating MOs: Started.\n')
        ## placeholder
        
        self.t_mos = self.get_t()

        utils.printer.write_out('Updating MOs: Done.\n')

        return
    
    
    def get_estate_energies(self, return_1d = True): ## placeholder
        """Get energy of each 'electronic state', which is i->a excitation configuration. Approximate state energy as MO energy difference."""

        return np.zeros_like(self.e_coeffs)

        #if not self.is_uptodate(self.t_estate_energies):

        #    utils.printer.write_out('Updating electronic state energies: Started.\n')
        #    
        #    mo_energies, mo_coeffs = self.get_molecular_orbitals()

        #    n_mo  = len(mo_energies)
        #    n_occ = self.n_occ
        #    n_vir = n_mo - self.n_occ
        #    
        #    if self.active_orbitals is None:
        #
        #        self.active_orbitals = [ True for i_mo in range(n_mo) ]
        #
        #    state_energies = []
        #
        #    for i_occ in range(n_occ):
        #        
        #        if self.active_orbitals[i_occ]:
        #
        #            row = []
        #
        #            for i_vir in range(n_vir):
        #
        #                if self.active_orbitals[i_vir + n_occ]:
        #
        #                    row.append( self.gs_energy + self.mo_energies[i_vir + n_occ] - self.mo_energies[n_mo - n_occ] )
        #
        #            state_energies.append(row)
        #    
        #    self.old_estate_energies = deepcopy(self.estate_energies)
        #    self.estate_energies     = np.array(state_energies)

        #    self.t_estate_energies = self.get_t()
        #    
        #    utils.printer.write_out('Updating electronic state energies: Done.\n')
        #
        #if return_1d:

        #    return np.sort( self.estate_energies.flatten() )

        #else:

        #    return deepcopy(self.estate_energies)


    def get_tdnac(self): ## placeholder
        
        n_estate = self.get_n_estate()

        old_position_2d = utils.coord_1d_to_2d(self.old_position)
        position_2d     = utils.coord_1d_to_2d(self.position)
        
        if self.qc_program == 'dftb+':

            overlap_twogeom = dftbplus_manager.worker.return_overlap_twogeom(old_position_2d, position_2d)

        else:

            utils.stop_with_error("Unknown quantum chemistry program %s .\n" % self.qc_program)

        print(overlap_twogeom) ## Debug code

        tdnac = np.zeros( (n_estate, n_estate), dtype='float64' )
        
        return tdnac

    
    def get_e_coeffs(self):
        return deepcopy(self.e_coeffs)


    def get_old_e_coeffs(self):
        return deepcopy(self.old_e_coeffs)


    def get_e_coeffs_tderiv(self):
        return deepcopy(self.e_coeffs_tderiv)


    def get_old_e_coeffs_tderiv(self):
        return deepcopy(self.old_e_coeffs_tderiv)


    def get_force(self): # placeholder
        """Get nuclear force originating from electronic states."""

        force = np.zeros_like(self.position)

        return force
    

    def update_matrices(self):

        if not self.is_uptodate(self.t_matrices):

            utils.printer.write_out('Updating hamiltonian and overlap matrices: Started.\n')

            if self.qc_program == 'dftb+':

                n_AO, self.H, self.S = dftbplus_manager.run_dftbplus_text(self.atomparams, self.position)

            else:
                
                utils.stop_with_error("Unknown quantum chemistry program %s .\n" % self.qc_program)

            self.t_matrices = self.get_t()

            utils.printer.write_out('Updating hamiltonian and overlap matrices: Done.\n')

        return


    def set_new_e_coeffs(self, e_coeffs):
        
        self.old_e_coeffs = self.e_coeffs
        self.e_coeffs = e_coeffs

        return


    def set_new_e_coeffs_tderiv(self, e_coeffs_tderiv):
        
        self.old_e_coeffs_tderiv = self.e_coeffs_tderiv
        self.e_coeffs_tderiv = e_coeffs_tderiv

        return

    
    def construct_initial_gs(self):

        if self.qc_program == 'dftb+':
        
            n_atom = len(self.atomparams)

            coords = self.position.reshape(n_atom, 3) * ANGST2AU
            
            dftbplus_manager.worker.set_geometry(coords)

            self.gs_energy = dftbplus_manager.worker.get_energy()

            self.mo_energies, mo_coeffs_real = dftbplus_manager.worker.get_molecular_orbitals(
                open_shell = self.is_open_shell
            )

            self.mo_coeffs = mo_coeffs_real.astype('complex128')

            self.gs_filling = dftbplus_manager.worker.get_filling(open_shell = self.is_open_shell)

            self.n_elec = np.sum(self.gs_filling)

            self.n_MO = np.size(self.mo_coeffs, 2)
            self.n_AO = self.n_MO

            self.update_gs_density_matrix()

        else:

            utils.stop_with_error("Not compatible with quantum chemistry program %s ." % self.qc_program)

        return


    def update_gs_density_matrix(self):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        self.gs_rho = np.zeros( (n_spin, self.n_MO, self.n_MO), dtype = 'float64' )

        # diagonal matrix whose elements are occupation numbers
        f = np.zeros( (self.n_MO, self.n_MO), dtype = 'float64' )

        for i_spin in range(n_spin):

            for i_MO in range(self.n_MO):

                f[i_MO,i_MO] = self.gs_filling[i_spin][i_MO]
            
            rho = np.real( np.dot( np.transpose(self.mo_coeffs[0,:,:]), np.dot(f, self.mo_coeffs[0,:,:]) ) )

            self.gs_rho[i_spin, :, :] = rho[:, :]
        
        return
