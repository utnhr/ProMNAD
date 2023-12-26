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
from electronmodule import Electronic_state
#from worldsmodule import World

class Tbf:
    """Class of TBFs."""


    @classmethod
    def get_gaussian_overlap(cls, tbf1, tbf2):
        """Calculate overlap matrix element (for each degree of freedom) between two gaussian wave packets."""

        pos1 = tbf1.get_position()
        pos2 = tbf2.get_position()
        mom1 = tbf1.get_momentum()
        mom2 = tbf2.get_momentum()

        delta_pos   = tbf2.position - tbf1.position
        delta_mom   = tbf2.momentum - tbf2.momentum
        delta_phase = tbf2.get_phase() - tbf1.get_phase()

        mid_pos   = 0.5 * (pos1 + pos2)
        
        vals = []

        n_dof = tbf1.get_n_dof()
        width = tbf1.get_width()

        for i_dof in range(n_dof):
            
            val = 1.0

            val *= exp( -0.5*width[i_dof]*delta_pos[i_dof]**2 )
            val *= exp( -delta_mom[i_dof]**2/(8*width[i_dof]*H_DIRAC**2) )
            val *= exp( (1j/H_DIRAC)*(mom1[i_dof]*pos1[i_dof] - \
                                      mom2[i_dof]*pos2[i_dof] + \
                                      mid_pos[i_dof]*delta_mom[i_dof])
                      )
            val *= exp( (1j/H_DIRAC)*delta_phase[i_dof] )

            vals.append(val)

        return np.array(vals)

    
    @classmethod
    def get_gaussian_dR(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate <x_m|d/dR|x_n>"""

        s_gw  = gaussian_overlap
        Pbar  = 0.5 * ( tbf1.get_momentum() + tbf2.get_momentum() )
        DelR  = tbf2.get_position() - tbf1.get_position()
        width = tbf1.get_width()

        return s_gw.prod() * ( (1.0j / H_DIRAC) * Pbar + width * DelR )


    @classmethod
    def get_gaussian_dP(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate <x_m|d/dP|x_n>"""

        s_gw  = gaussian_overlap
        DelP  = tbf2.get_momentum() - tbf1.get_momentum()
        DelR  = tbf2.get_position() - tbf1.get_position()
        width = tbf1.get_width()

        return s_gw.prod() * (
            ( -1.0 / (4.0 * width * H_DIRAC**2) ) * DelP - \
            ( 1.0j / (2.0 * H_DIRAC) ) * DelR
        )


    @classmethod
    def get_wf_overlap(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate overlap matrix between two TBFs."""
        s_gw = gaussian_overlap
        return s_gw.prod() * np.dot(
            tbf1.e_part.get_e_coeffs(), tbf2.e_part.get_e_coeffs()
        )

    
    @classmethod
    def get_gaussian_kinE_term(cls, tbf1, tbf2, gaussian_overlap, each_degree=False):
        """Calculate kinetic energy matrix element between two Gaussian wave packets. Integrate over all degrees of freedom."""

        #if gaussian_overlap = None:
        #    s_gw = cls.get_gaussian_overlap(tbf1, tbf2)
        #else:
        #    s_gw = gaussian_overlap

        s_gw  = gaussian_overlap
        width = tbf1.get_width()
        del_R = tbf2.get_position() - tbf2.get_position()
        mid_P = 0.5 * ( tbf1.get_momentum() + tbf2.get_momentum() )
        mass  = tbf1.get_mass()

        kin = - 0.5 * H_DIRAC**2 * (
            (1j/H_DIRAC) * 2.0 * width * del_R * mid_P - \
            width + width**2 * del_R**2 - mid_P**2 / H_DIRAC**2
        ) * s_gw / mass 

        if each_degree:
            return kin
        else:
            return kin.prod()
    
    
    @classmethod
    def get_gaussian_potential_term(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate potential matrix element between two Gaussian wave packets. Electronic state energies must be calculated in advance."""

        n_estate = tbf1.get_n_estate()

        estate_energies_1 = tbf1.e_part.get_estate_energies()
        estate_energies_2 = tbf2.e_part.get_estate_energies()

        return 0.5 * gaussian_overlap.prod() * (
            estate_energies_1 + estate_energies_2
        ) # array, length = number of electronic states


    @classmethod
    def get_gaussian_NAcoupling_term(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate nonadiabatic coupling matrix element between two Gaussian wave packets."""
        return (1j * 0.5 / H_DIRAC) * gaussian_overlap.prod() * (
            tbf1.e_part.get_tdnac() + tbf2.e_part.get_tdnac()
        )


    @classmethod
    def get_tbf_hamiltonian_element_BAT(cls, tbf1, tbf2, gaussian_overlap=None):
        """Calculate hamiltonian matrix elements between TBFs based on bra-ket avaraged Taylor expansion (BAT)."""

        if gaussian_overlap is None:

            gaussian_overlap = cls.get_gaussian_overlap(tbf1, tbf2)

        n_estates = ( tbf1.get_n_estate(), tbf2.get_n_estate() )

        e_coeffs = ( tbf1.e_part.get_e_coeffs(), tbf2.e_part.get_e_coeffs() )

        kinE       = cls.get_gaussian_kinE_term(tbf1, tbf2, gaussian_overlap, each_degree = False)
        potentials = cls.get_gaussian_potential_term(tbf1, tbf2, gaussian_overlap)
        tdnacs     = cls.get_gaussian_NAcoupling_term(tbf1, tbf2, gaussian_overlap)

        vals = [ [ 0.0j for i_estate in range(n_estates[0]) ] for j_estate in range(n_estates[1]) ]

        for i_estate in range(n_estates[0]):
            for j_estate in range(n_estates[1]):

                fac = e_coeffs[0][i_estate] * e_coeffs[1][j_estate]

                val = 0.0j

                if i_estate == j_estate:

                    val += kinE
                    val += potentials[i_estate]

                val += tdnacs[i_estate, j_estate]

        return fac * val


    @classmethod
    def get_gaussian_derivative_coupling(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate <xi_m|d/dt|xi_n>, where xi_m(n) is a GWP"""

        momentum = tbf2.get_momentum()
        velocity = tbf2.get_velocity()
        dgdt     = 0.5 * np.dot(momentum, velocity) # dgamma/dt; time-dependence of phase

        term1 = np.dot( velocity, cls.get_gaussian_dR(tbf1, tbf2, gaussian_overlap) )
        term2 = np.dot( momentum, cls.get_gaussian_dP(tbf1, tbf2, gaussian_overlap) )
        term3 = (1.0j/H_DIRAC) * dgdt * gaussian_overlap.prod()

        return term1 + term2 + term3

    
    @classmethod
    def get_tbf_derivative_coupling(cls, tbf1, tbf2, gaussian_overlap):
        """Calculate <\psi_m|d/dt|\psi_n>, where \psi_m(n) is a TBF"""

        gwp_derivative = cls.get_gaussian_derivative_coupling(
            tbf1, tbf2, gaussian_overlap
        )
        
        e_coeffs_1        = tbf1.e_part.get_e_coeffs()
        e_coeffs_2        = tbf2.e_part.get_e_coeffs()

        e_coeffs_tderiv_2 = tbf2.e_part.get_e_coeffs_tderiv()

        term1 = gwp_derivative * np.dot( np.conjugate(e_coeffs_1), e_coeffs_2)
        term2 = gaussian_overlap.prod() * np.dot( np.conjugate(e_coeffs_1), e_coeffs_tderiv_2 )

        return term1 + term2


    def __init__(
        self, settings, atomparams, position, n_dof, n_estate, tbf_id,
        momentum=None, mass=None, width=None, phase=None, e_coeffs=None, initial_gs_energy=None, t=0,
        ):

        self.atomparams    = atomparams        # dictionary { 'elems': array, 'angmom_table': array (optional, for DFTB+) }
        self.n_dof         = n_dof             # integer
        self.n_estate      = n_estate          # integer
        self.init_position = position          # np.array (n_dof)
        self.mass          = mass              # np.array (n_dof)
        self.width         = width             # np.array (n_dof)
        self.init_t        = t                 # integer, index of step
        self.tbf_id        = tbf_id            # integer (start 0)
        #self.world_id      = world_id          # int (start 0)

        if momentum is not None:
            self.init_momentum = momentum # np.array (n_dof)
        else:
            self.init_momentum = np.zeros(n_dof, dtype='float64')

        if phase is not None:
            self.phase = phase # np.array (n_dof)
        else:
            self.phase = np.zeros(n_dof, dtype='float64')

        self.gs_energy     = initial_gs_energy # None or float

        self.position     = deepcopy(self.init_position)
        self.old_position = deepcopy(self.position)
        self.t_position   = t

        self.momentum     = deepcopy(self.init_momentum)
        self.old_momentum = deepcopy(self.momentum)
        self.t_momentum   = t

        self.force        = np.zeros_like(self.position)
        self.old_force    = np.zeros_like(self.position)

        self.t = self.init_t

        self.e_dot = np.zeros(n_estate) # dc/dt

        #self.tbf_id = world.total_tbf_count + 1
        
        self.e_part = Electronic_state(
            settings, atomparams, e_coeffs, self.get_position(), self.get_velocity(), self.t
        )
        #self.e_part.set_new_position_velocity_time(
        #    self.get_position(), self.get_velocity(), self.get_t()
        #)

        self.is_alive = True

        #self.world = World.worlds[self.world_id]
        #self.world.add_tbf(self)

        return


    def destroy(self):

        self.is_alive = False
        self.world.destroy_tbf(self)

        print( "World %d: TBF %d destroyed.\n" % (self.world_id, self.tbf_id) )

        return


    def get_n_dof(self):
        return self.n_dof


    def get_n_estate(self):
        return self.n_estate


    def get_mass(self):
        return deepcopy(self.mass)


    def get_width(self):
        return deepcopy(self.width)


    #def get_e_coeffs(self):
    #    return deepcopy(self.e_coeffs)


    #def get_tdnac(self):
    #    return deepcopy(self.e_part.get_tdnac())


    def get_position(self):
        return deepcopy(self.position)

    
    def get_old_position(self):
        return deepcopy(self.old_position)


    def get_momentum(self):
        return deepcopy(self.momentum)


    def get_old_momentum(self):
        return deepcopy(self.old_momentum)


    def get_velocity(self):
        return deepcopy(self.momentum / self.mass)


    def get_old_velocity(self):
        return deepcopy(self.old_momentum / self.mass)


    def get_phase(self):
        return deepcopy(self.phase)


    def get_t(self):
        return self.t


    def get_n_estate(self):
        return self.n_estate


    def get_tbf_id(self):
        return self.tbf_id


    def get_atominfo(self):
        return deepcopy(self.atominfo)

    
    def get_nuc_wf_val(self, coord):
        """Get the value of nuclear wavefunction (gaussian wave packet) at a given nuclear coordinate."""

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

        return val


    #def propagate_indivisual_tbf(self):

    #    ## placeholder
    #    propagate_tbf(self)

    #    self.e_part.update_position_velocity_time(
    #        self.get_position(), self.get_velocity(), self.get_t()
    #    )
    #    
    #    return

    
    def spawn(self):
        
        ## placeholder
        
        return


    def set_new_position(self, position, e_part_too=False):
        
        self.old_position = self.position
        self.position     = deepcopy(position)

        if e_part_too:
            self.e_part.set_new_position(self.position)

        return


    def set_new_momentum(self, momentum, e_part_too=False):
        
        self.old_momentum = self.momentum
        self.momentum     = deepcopy(momentum)

        if e_part_too:
            self.e_part.set_new_momentum(self.momentum)

        return


    def update_force(self):
        
        self.old_force = self.force
        self.force     = self.e_part.get_force()

        return


    def get_force(self):
        return deepcopy(self.force)


    def set_new_time(self, t, e_part_too=False):
        
        self.t = t

        if e_part_too:
            self.e_part.set_new_t(self.t)

        return


    def update_electronic_part(self, dt):

        # update electronic coeffs (leapfrog)
        
        old_e_coeffs        = self.e_part.get_old_e_coeffs()
        e_coeffs_tderiv     = self.e_part.get_e_coeffs_tderiv()

        e_coeffs = old_e_coeffs + 2.0 * e_coeffs_tderiv * dt

        self.e_part.set_new_e_coeffs(e_coeffs)

        # update position and velocity for electronic part

        t = self.get_t()

        self.e_part.set_new_position( self.get_position() )
        self.e_part.set_new_momentum( self.get_momentum() )
        self.e_part.set_new_time( self.get_t() )

        # update electronic wavefunc and relevant physical quantities

        self.e_part.update_matrices()

        estate_energies = self.e_part.get_estate_energies()

        tdnac = self.e_part.get_tdnac()
        
        # construct electronic Hamiltonian
        # and update time derivative of electronic coeffs

        n_estate = self.e_part.get_n_estate()

        H_el = -1.0j * H_DIRAC * tdnac
        for i_estate in range(n_estate):
            H_el[i_estate,i_estate] += estate_energies[i_estate]

        e_coeffs_tderiv = (-1.0j / H_DIRAC) * np.dot(H_el, e_coeffs)

        self.e_part.set_new_e_coeffs_tderiv(e_coeffs_tderiv)

        self.set_new_time( self.get_t() + 1 )

        return


    def update_position_and_velocity(self, dt):
        
        old_position = self.get_old_position()
        velocity     = self.get_velocity()

        position = old_position + 2.0 * velocity * dt

        old_momentum = self.get_old_momentum()
        force        = self.get_force()

        momentum = old_momentum + 2.0 * force * dt

        self.set_new_position(position)
        self.set_new_momentum(momentum)

        return

