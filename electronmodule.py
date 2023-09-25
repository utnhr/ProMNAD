#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy as sp
from copy import deepcopy

from constants import H_DIRAC
import utils

class electronic_structure:
    

    @classmethod
    def set_common_electronic_structure_options(cls, n_occ, active_orbitals, qc_program):
        
        cls.n_occ           = n_occ
        cls.active_orbitals = active_orbitals
        cls.qc_program      = qc_program


    def __init__(self):
        
        self.H               = None
        self.S               = None
        self.mo_energies     = None
        self.mo_coeffs       = None
        self.n_occ           = None
        self.gs_energy       = None
        self.active_orbitals = None
        self.atominfo        = None

        self.t_molecular_orbitals = float('inf')
        self.t_estate_energies    = float('inf')
        self.t_matrices           = float('inf')
        self.t_force              = float('inf')
        self.t_tdnac              = float('inf')
        self.initial_MO_done      = False

        return


    def update_position_velocity_time(self, position, velocity, t):
        
        self.position = position
        self.velocity = velocity
        self.t        = t

        return


    def is_uptodate(self, time_last):
        
        if util.is_equal_scalar(self.get_t(), time_last):
            return True
        else:
            return False


    def get_t(self):
        return self.t

    
    def get_n_occ(self):
        return self.n_occ


    def get_gs_energy(self):
        return self.gs_energy


    def get_active_orbitals(self):
        return deepcopy(self.active_orbitals)


    def get_atominfo(self):
        return deepcopy(self.atominfo)


    def get_H(self):

        if not electronic_structure.is_uptodate(self.t_matrices):
            self.update_matrices()

        return deepcopy(self.H)


    def get_S(self):

        if not electronic_structure.is_uptodate(self.t_matrices):
            self.update_matrices()

        return deepcopy(self.S)


    def get_molecular_orbitals(self):
        """Solve F * C = S * C * e ; get MO energies and coefficients."""    
        if not self.initial_MO_done:
            self.construct_initial_molecular_orbitals()

        elif not electronic_structure.is_uptodate(self.t_molecular_orbitals):
            self.update_molecular_orbitals()

        return deepcopy(self.mo_energies), deepcopy(self.mo_coeffs)


    def construct_initial_molecular_orbitals(self):
        """Solve F * C = S * C * e ; get MO energies and coefficients."""    
        if not electronic_structure.is_uptodate(self.t_molecular_orbitals):

        H = self.get_H()
        S = self.get_S()

        self.mo_energies, self.mo_coeffs = sp.linalg.eigh(H, S, type=1)

        self.t_mos = self.get_t()

        return


    def update_molecular_orbitals(self):
        """Update MO energies and coefficients according to TD-KS equation."""
        utils.printer.write_out('Updating MOs: Started.')
        ## placeholder
        
        self.t_mos = self.get_t()

        utils.printer.write_out('Updating MOs: Done.')

        return
    
    
    def get_estate_energies(self):
        """Get energy of each 'electronic state', which is i->a excitation configuration. Approximate state energy as MO energy difference."""
        if not electronic_structure.is_uptodate(self.t_estate_energies):

            utils.printer.write_out('Updating electronic state energies: Started.')
            
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
            
            self.estate_energoes = np.array(state_energies)

            self.t_estate_energies = self.time
            
            utils.printer.write_out('Updating electronic state energies: Done.')
        
        return deepcopy(self.estate_energies)


    def get_tdnac(self):
        
        return None ## placeholder
    
    
    def get_force(self):
        """Get nuclear force originating from electronic states."""
        return None # placeholder
    

    def update_matrices(self):

        if not electronic_structure.is_uptodate(self.t_matrices):

            utils.printer.write_out('Updating hamiltonian and overlap matrices: Started.')

            if electronic_structure.qc_program == 'dftb+':

                n_AO, self.H, self.S = dftbplus_manager.run_dftbplus_text(self.atominfo['elems'], self.atominfo['angmom_table'], self.position)

            else:
                
                utils.stop_with_error("Unknown quantum chemistry program %s ." % electronic_structure.qc_program)

            self.t_matrices = self.get_t()

            utils.printer.write_out('Updating hamiltonian and overlap matrices: Done.')

        return
