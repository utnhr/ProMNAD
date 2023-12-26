#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy.linalg as sp
from copy import deepcopy

from constants import H_DIRAC
import utils

from interface_dftbplus import dftbplus_manager

class Electronic_state:
    

    def __init__(self, settings, atomparams, e_coeffs, position, velocity, t):

        self.n_occ               = settings['n_occ']
        self.active_occ_mos      = settings['active_occ_mos']
        self.active_vir_mos      = settings['active_vir_mos']
        self.qc_program          = settings['engine']['type']
        
        self.e_coeffs            = e_coeffs # np.array (n_estate)
        self.old_e_coeffs        = deepcopy(self.e_coeffs)
        self.estate_energies     = None
        self.old_estate_energies = None
        self.position            = position
        self.velocity            = velocity
        self.H                   = None
        self.S                   = None
        self.mo_energies         = None
        self.mo_coeffs           = None
        self.gs_energy           = 0.0 # placeholder
        self.active_orbitals     = None
        self.atomparams          = deepcopy(atomparams)

        self.e_coeffs_tderiv     = np.zeros_like(e_coeffs)
        self.old_e_coeffs_tderiv = np.zeros_like(e_coeffs)
        
        self.t_e_coeffs           = t
        self.t_molecular_orbitals = sys.maxsize
        self.t_estate_energies    = sys.maxsize
        self.t_matrices           = sys.maxsize
        self.t_force              = sys.maxsize
        self.t_tdnac              = sys.maxsize
        self.t_state_coeffs       = sys.maxsize

        self.initial_MO_done      = False

        return


    def set_new_position(self, position):
        self.position = deepcopy(position)
        return

    def set_new_momentum(self, momentum):
        self.momentum = deepcopy(momentum)
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


    def get_active_orbitals(self):
        return deepcopy(self.active_orbitals)


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
        if not self.initial_MO_done:
            self.construct_initial_molecular_orbitals()
            self.initial_MO_done = True

        elif self.is_uptodate(self.t_molecular_orbitals):
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
    
    
    def get_estate_energies(self, return_1d = True):
        """Get energy of each 'electronic state', which is i->a excitation configuration. Approximate state energy as MO energy difference."""
        if not self.is_uptodate(self.t_estate_energies):

            utils.printer.write_out('Updating electronic state energies: Started.\n')
            
            mo_energies, mo_coeffs = self.get_molecular_orbitals()

            n_mo  = len(mo_energies)
            n_occ = self.n_occ
            n_vir = n_mo - self.n_occ
            
            if self.active_orbitals is None:
        
                self.active_orbitals = [ True for i_mo in range(n_mo) ]
        
            state_energies = []
        
            for i_occ in range(n_occ):
                
                if self.active_orbitals[i_occ]:
        
                    row = []
        
                    for i_vir in range(n_vir):
        
                        if self.active_orbitals[i_vir + n_occ]:
        
                            row.append( self.gs_energy + self.mo_energies[i_vir + n_occ] - self.mo_energies[n_mo - n_occ] )
        
                    state_energies.append(row)
            
            self.old_estate_energies = deepcopy(self.estate_energies)
            self.estate_energies     = np.array(state_energies)

            self.t_estate_energies = self.get_t()
            
            utils.printer.write_out('Updating electronic state energies: Done.\n')
        
        if return_1d:

            return np.sort( self.estate_energies.flatten() )

        else:

            return deepcopy(self.estate_energies)


    def get_tdnac(self): ## placeholder
        
        n_estate = self.get_n_estate()
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
                
                utils.stop_with_error("Unknown quantum chemistry program %s .\n" % electronic_structure.qc_program)

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
