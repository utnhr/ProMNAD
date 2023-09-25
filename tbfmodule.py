#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy as sp
from copy import deepcopy

from constants import H_DIRAC
import utils
from interface_dftbplus import dftbplus_manager
import electronmodule

class tbf:
    """Class of TBFs."""
    
    # default calculation settings
    qc_program      = 'dftb+'
    
    # TBF group info
    total_tbf_count = 0
    live_tbf_count  = 0
    live_tbfs       = []


    @classmethod
    def init_calculation_settings(cls, qc_program=None, n_occ=None, active_orbitals=None, dt=None):
    #{{{
        if qc_program is not None:
            cls.qc_program = qc_program

        if n_occ is not None:
            cls.n_occ = n_occ

        if active_orbitals is not None:
            cls.active_orbitals = active_orbitals

        if dt is not None:
            cls.dt = dt
    #}}}
        return

    
    @classmethod
    def propagate_all(cls):
        """Time-propagate full wavefunction."""
        ## placeholder
        return


    @classmethod
    def get_gaussian_overlap(cls, tbf1, tbf2):
        """Calculate overlap matrix element (for each degree of freedom) between two gaussian wave packets."""
    #{{{
        #if tbf1.get_n_dof() != tbf2.get_n_dof():
        #    utils.stop_with_error('Number of electronic states must be the same for all TBFs')
        #n_dof = tbf1.get_n_dof()

        #if not utils.is_equal_ndarray(tbf1.width, tbf2.width):
        #    utils.stop_with_error('Gaussian width must be the same for all TBFs')
        #width = tbf1.get_width()

        pos1 = tbf1.get_position()
        pos2 = tbf2.get_position()
        mom1 = tbf1.get_momentum()
        mom2 = tbf2.get_momentum()

        delta_pos   = tbf2.position - tbf1.position
        delta_mom   = tbf2.momentum - tbf2.momentum
        delta_phase = tbf2.get_phase() - tbf1.get_phase()

        mid_pos   = 0.5 * (pos1 + pos2)
        
        vals = []

        for i_dof in range(n_dof):
            
            val = 1.0

            val *= exp( -0.5*width[i_dof]*delta_pos[i_dof]**2 )
            val *= exp( -delta_mom[i_dof]**2/(8*width[i_dof]*H_DIRAC**2) )
            val *= exp( (1j/H_DIRAC)*(momentum[i_dof])**2 )
            val *= exp( (1j/H_DIRAC)*(mom1[i_dof]*pos1[i_dof] - mom2[i_dof]*pos2[i_dof] + mid_pos[i_dof]*delta_mom[i_dof]) )
            val *= exp( (1j/H_DIRAC)*delta_phase[i_dof] )

            vals.append(val)
    #}}}
        return np.array(vals)


    @classmethod
    def get_wf_overlap(cls, tbf1, tbf2, gaussian_overlap=None):
        """Calculate overlap matrix between two TBFs."""
    #{{{
        if gaussian_overlap is None:
            s_gw = cls.get_gaussian_overlap(tbf1, tbf2)
        else:
            s_gw = gaussian_overlap
    #}}}
        return s_gw * np.dot( tbf1.get_e_coeffs(), tbf2.get_e_coeffs() )

    
    @classmethod
    def get_gaussian_kinE_term(cls, tbf1, tbf2, gaussian_overlap, each_degree=False):
        """Calculate kinetic energy matrix element between two Gaussian wave packets. Integrate over all degrees of freedom."""
    #{{{
        #if gaussian_overlap = None:
        #    s_gw = cls.get_gaussian_overlap(tbf1, tbf2)
        #else:
        #    s_gw = gaussian_overlap

        s_gw  = gaussian_overlap
        width = tbf1.get_width()
        del_R = tbf2.get_position() - tbf2.get_position()
        mid_P = 0.5 * ( tbf1.get_momentum() + tbf2.get_momentum() )
        mass  = tbf1.get_mass

        kin = - 0.5 * H_DIRAC**2 * (
            (1j/H_DIRAC) * 2.0 * width * del_R * mid_P - width + width**2 * del_R**2 - mid_P**2 / H_DIRAC**2
        ) * s_gw / mass 
    #}}}
        if each_degree:
            return kin
        else:
            return kin.prod()
    
    
    @classmethod
    def get_gaussian_potential_term(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate potential matrix element between two Gaussian wave packets. Electronic state energies must be calculated in advance."""
    #{{{
        n_estate = tbf1.get_n_estate()

        estate_energies_1 = tbf1.get_estate_energies()
        estate_energies_2 = tbf2.get_estate_energies()
    #}}}
        return 0.5 * gaussian_overlap.prod() * (estate_energies_1 + estate_energies_2) # array, length = number of electronic states


    @classmethod
    def get_gaussian_NAcoupling_term(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate nonadiabatic coupling matrix element between two Gaussian wave packets."""
        return (1j * 0.5 / H_DIRAC) * ( tbf1.get_tdnac() + tbf2.get_tdnac() )


    @classmethod
    def get_tbf_hamiltonian_element_BAT(cls, tbf1, tbf2, gaussian_overlap=None):
        """Calculate hamiltonian matrix elements between TBFs based on bra-ket avaraged Taylor expansion (BAT)."""
    #{{{
        if gaussian_overlap = None:

            gaussian_overlap = cls.get_gaussian_overlap(tbf1, tbf2)

        n_estates = ( tbf1.get_n_estate(), tbf2.get_n_estate() )

        vals = [ [ 0.0j for i_estate in range(n_estates[0]) ] for j_estate in range(n_estates[1]) ]

        for i_estate in range(n_estates[0]):
            for j_estate in range(n_estates[0], n_estates[1]):

                val = 0.0j

                if i_estate == j_estate:

                    val += cls.get_gaussian_kinE_term(tbf1, tbf2, gaussian_overlap)
                    val += cls.get_gaussian_potential_term(tbf1, tbf2, gaussian_overlap)
                
                val += cls.get_gaussian_NAcoupling_term(tbf1, tbf2, gaussian_overlap)

            vals[i_estate][j_estate] = val
            vals[j_estate][i_estate] = val
    #}}}
        return sum(vals)


    def __init__(
        self, atominfo, position, n_dof, n_estate, initial_gs_energy, momentum=None, mass=None, width=None, phase=None, e_coeffs=None, t=0,
    ):
    #{{{
        self.atominfo      = atominfo          # dictionary { 'elems': array, 'angmom_table': array (optional, for DFTB+) }
        self.n_dof         = n_dof             # integer
        self.n_estate      = n_estate          # integer
        self.init_position = position          # np.array (n_dof)
        self.mass          = mass              # np.array (n_dof)
        self.width         = width             # np.array (n_dof)
        self.e_coeffs      = e_coeffs          # np.array (n_estate)
        self.init_t        = t                 # float
        self.gs_energy     = initial_gs_energy # float

        if momentum is not None:
            self.init_momentum = momentum # np.array (n_dof)
        else:
            self.init_momentum = np.zeros(n_dof)

        if phase is not None:
            self.phase = phase # np.array (n_dof)
        else:
            self.phase = np.zeros(n_dof)

        self.position = deepcopy(self.init_position)
        self.momentum = deepcopy(self.init_momentum)
        self.t        = self.init_t

        self.e_dot    = np.zeros(n_estate) # dc/dt

        self.tbf_id   = tbf.total_tbf_count + 1
        
        self.e_part = electronmodule.electronic_state()
        self.e_part.update_position_velocity_time(
            self.get_position(), self.get_velocity(), self.get_t()
        )

        tbf.live_tbfs.append(self)
        tbf.total_tbf_count += 1
        tbf.live_tbf_count  += 1
    #}}}
        return

    def destroy(self):
    #{{{
        for i_live_tbf, live_tbf in enumerate(tbf.live_tbfs):
            
            if live_tbf.tbf_id == self.tbf_id:

                my_tbf_index = i_live_tbf
                
                break

        tbf.live_tbfs.pop(my_tbf_index)
        tbf.live_tbf_count -= 1

        print("TBF %d destroyed.\n" % self.tbf_id)
    #}}}
        return


    def get_n_dof(self):
        return self.n_dof


    def get_n_estate(self):
        return self.n_estate


    def get_mass(self):
        return deepcopy(self.mass)


    def get_width(self):
        return deepcopy(self.width)


    def get_e_coeffs(self):
        return deepcopy(self.e_coeffs)


    def get_position(self):
        return deepcopy(self.position)


    def get_momentum(self):
        return deepcopy(self.momentum)


    def get_t(self):
        return self.t


    def get_velocity(self):
        return deepcopy(self.momentum / self.mass)


    def get_phase(self):
        return deepcopy(self.phase)


    def get_n_estate(self):
        return self.n_estate


    def get_tbf_id(self):
        return self.tbf_id


    def get_atominfo(self):
        return deepcopy(self.atominfo)


    def get_nuc_wf_val(self, coord):
        """Get the value of nuclear wavefunction (gaussian wave packet) at a given nuclear coordinate."""
    #{{{
        n_dof    = self.get_n_dof()
        position = self.get_position()
        momentum = self.get_momentum()

        delta_r  = coord - position

        phase    = self.get_phase()
        
        val = 1.0
        
        for i_dof in range(self.get_n_dof()):

            val *= (2 * width[i_dof] / cmath.pi) ** (n_dof / 4)
            val *= exp( -width[i_dof] * delta_r[i_dof]**2 )
            val *= exp( -(1j/H_DIRAC) * momentum[i_dof] * delta_r[i_dof] )
            val *= exp( -(1j/H_DIRAC) * phase[i_dof] ) / 2
    #}}}    
        return val


    def propagate_indivisual_tbf(self):

        ## placeholder

        self.e_part.update_position_velocity_time(
            self.get_position(), self.get_velocity(), self.get_t()
        )
        
        return

    
    def spawn(self):
        
        ## placeholder
        
        return
