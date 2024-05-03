#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy.linalg as sp
from copy import deepcopy

from constants import H_DIRAC, AMU2AU, AU2ANGST, AU2EV, KB_EV, AU2SEC
import utils
from interface_dftbplus import dftbplus_manager
from electronmodule import Electronic_state
from settingsmodule import load_setting
from integratormodule import Integrator
from files import GlobalOutputFiles, LocalOutputFiles
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

        #delta_pos   = tbf2.position - tbf1.position
        #delta_mom   = tbf2.momentum - tbf1.momentum
        delta_pos   = pos2 - pos1
        delta_mom   = mom2 - mom1
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

        #print('VALS', vals) ## Debug code

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
        
        #print('OVERLAP_GWP_PART',s_gw.prod()) ## Debug code
        #print('OVERLAP_E_PART', np.dot(np.conj(tbf1.e_part.get_e_coeffs()), tbf2.e_part.get_e_coeffs()) ) ## Debug code

        return s_gw.prod() * np.dot(
            np.conj(tbf1.e_part.get_e_coeffs()), tbf2.e_part.get_e_coeffs()
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
            #$return kin.prod()
            return kin.sum()
    
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

        #vals = [ [ 0.0j for i_estate in range(n_estates[0]) ] for j_estate in range(n_estates[1]) ]

        #print('KINE', kinE) ## Debug code
        #print('POTENTIALS', potentials) ## Debug code

        H_mn = 0.0j

        for i_estate in range(n_estates[0]):
            for j_estate in range(n_estates[1]):

                fac = e_coeffs[0][i_estate].conj() * e_coeffs[1][j_estate]

                val = 0.0j

                if i_estate == j_estate:

                    val += kinE
                    val += potentials[i_estate]

                val += tdnacs[i_estate, j_estate]

                H_mn += fac * val

        return H_mn


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
        momentum=None, mass=None, width=None, phase=None, e_coeffs=None, initial_gs_energy=None, istep=0,
        is_fixed=False,
        ):

        self.atomparams    = atomparams        # dictionary { 'elems': array, 'angmom_table': array (optional, for DFTB+) }
        self.n_dof         = n_dof             # integer
        self.n_estate      = n_estate          # integer
        self.init_position = position          # np.array (n_dof)
        self.mass          = mass              # np.array (n_dof)
        self.width         = width             # np.array (n_dof)
        self.init_istep    = istep             # integer, index of step
        self.tbf_id        = tbf_id            # integer (start 0)
        self.is_fixed      = is_fixed          # logical
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

        self.istep = self.init_istep

        self.position       = deepcopy(self.init_position)
        self.old_position   = deepcopy(self.position)
        self.istep_position = istep

        self.momentum       = deepcopy(self.init_momentum)
        self.old_momentum   = deepcopy(self.momentum)
        self.istep_momentum = istep

        self.dt = load_setting(settings, 'dt')
        self.read_traject = load_setting(settings, 'read_traject')
        self.print_xyz_interval = load_setting(settings, 'print_xyz_interval')
        self.flush_interval = load_setting(settings, 'flush_interval')
        self.integmethod = load_setting(settings, 'integrator')

        if self.read_traject:

            given_geoms      = load_setting(settings, 'given_geoms')
            given_velocities = load_setting(settings, 'given_velocities')

            self.given_geoms      = deepcopy(given_geoms)
            self.given_velocities = deepcopy(given_velocities)

            position, velocity = self.get_next_given_traject()

            self.position = position
            self.momentum = self.mass * velocity
            self.old_position = deepcopy(self.position)
            self.old_momentum = deepcopy(self.momentum)

            self.istep = self.init_istep # istep has been updated in get_next_given_traject(), but is should be set to 0

        self.force        = np.zeros_like(self.position)
        self.old_force    = np.zeros_like(self.position)
        self.gs_force     = np.zeros_like(self.position)
        self.old_gs_force = np.zeros_like(self.position)

        self.e_dot = np.zeros(n_estate) # dc/dt

        #self.tbf_id = world.total_tbf_count + 1
        
        self.e_part = Electronic_state(
            settings, atomparams, e_coeffs, self.get_position(), self.get_velocity(), settings['dt'], self.istep,
            construct_initial_gs = True,
        )
        #self.e_part.set_new_position_velocity_time(
        #    self.get_position(), self.get_velocity(), self.get_t()
        #)

        self.is_alive = True

        #self.world = World.worlds[self.world_id]
        #self.world.add_tbf(self)

        self.Epot    = 0.0
        self.EpotGS  = 0.0
        self.dEpotGS = 0.0
        self.Ekin    = 0.0
        self.t       = self.dt * self.istep

        self.integrator = Integrator(self.integmethod)
        self.phase_integrator = Integrator('adams_bashforth_2')

        self.localoutput = LocalOutputFiles(self.tbf_id)

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


    def get_mass_au(self):
        return deepcopy(self.mass) * AMU2AU


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
        return deepcopy(self.momentum / self.get_mass_au())


    def get_old_velocity(self):
        return deepcopy(self.old_momentum / self.get_mass_au())


    def get_phase(self):
        return deepcopy(self.phase)


    def get_istep(self):
        return self.istep


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
            self.e_part.set_new_velocity(self.get_velocity())

        return


    def update_force(self):
        
        self.old_force = self.force
        self.force     = self.e_part.get_force()

        self.old_gs_force = self.gs_force
        self.gs_force     = self.e_part.get_force(gs_force = True)

        return


    def get_force(self):
        return deepcopy(self.force)


    def set_new_istep(self, istep, e_part_too=False):
        
        self.istep = istep

        if e_part_too:
            self.e_part.set_new_istep(self.istep)

        return


    def make_e_coeffs_tderiv(self, t, e_coeffs, H_el):

        e_coeffs_tderiv = (-1.0j / H_DIRAC) * np.dot(H_el, e_coeffs)

        return e_coeffs_tderiv


    def update_electronic_part(self, dt):

        # update position and velocity for electronic part

        self.e_part.set_next_position( self.get_position() )
        self.e_part.set_next_velocity( self.get_velocity() )
        #self.e_part.set_next_istep( self.get_istep() )

        self.e_part.update_position()
        self.e_part.update_velocity()

        # update electronic wavefunc and relevant physical quantities

        self.e_part.update_matrices()

        self.e_part.update_estate_energies()

        estate_energies = self.e_part.get_estate_energies()
        
        # origin of electronic state energies

        if self.e_part.initial_estate_energies is None:
            self.e_part.initial_estate_energies = [ estate_energies[0] for i_state in range( len(estate_energies) ) ]

        estate_energies -= self.e_part.initial_estate_energies

        self.e_part.update_tdnac()

        tdnac = self.e_part.get_tdnac()
        
        # construct electronic Hamiltonian
        # and update time derivative of electronic coeffs

        n_estate = self.e_part.get_n_estate()

        H_el = -1.0j * H_DIRAC * tdnac
        for i_estate in range(n_estate):
            H_el[i_estate,i_estate] += estate_energies[i_estate]

        e_coeffs = self.e_part.get_e_coeffs()

        t = dt * self.get_istep()

        e_coeffs = self.integrator.engine(dt, t, e_coeffs, self.make_e_coeffs_tderiv, H_el)

        self.e_part.set_new_e_coeffs(e_coeffs)

        e_coeffs_tderiv = self.make_e_coeffs_tderiv(t, e_coeffs, H_el)

        self.e_part.set_new_e_coeffs_tderiv(e_coeffs_tderiv)

        self.print_results()

        self.set_new_istep( self.get_istep() + 1 )

        self.t += self.dt

        return


    def dgdt(self, t, phase): # dgamma/dt; time-dependence of phase

        momentum = self.get_momentum()
        velocity = self.get_velocity()
        
        return 0.5 * np.dot(momentum, velocity)


    def update_position_and_velocity(self, dt):
        
        old_position = self.get_old_position()
        velocity     = self.get_velocity()

        if self.read_traject:

            position, velocity = self.get_next_given_traject()

        else:

            #position = old_position + 2.0 * velocity * dt
            position = self.get_position() + velocity * dt + 0.5 * ( self.force / self.get_mass_au() ) * dt**2

        old_momentum = self.get_old_momentum()
        
        self.update_force()
        #force = self.get_force()

        if self.read_traject:
                
            momentum = velocity * self.get_mass_au()

        else:

            #momentum = old_momentum + 2.0 * force * dt
            momentum = self.get_momentum() + 0.5 * (self.old_force + self.force) * dt

        self.phase = self.phase_integrator.engine(dt, 0, self.phase, self.dgdt)

        if self.is_fixed:
            position = old_position
            momentum = old_momentum

        delta = 0.5 * (position - old_position)
        self.Epot   -= np.dot(delta, self.force)
        self.dEpotGS = -np.dot(delta, self.gs_force)
        self.Ekin    = sum( 0.5 * momentum**2 / self.get_mass_au() )
        
        self.EpotGS += self.dEpotGS
        self.e_part.modify_gs_energy(self.dEpotGS)

        self.set_new_position(position)
        self.set_new_momentum(momentum)

        veloc = self.get_velocity()

        return


    def get_temperature(self):
        
        e_kin_per_dof_ev = AU2EV * self.Ekin / np.size(self.position)
        temp_kelvin = e_kin_per_dof_ev / (0.5 * KB_EV)
        return temp_kelvin

    
    def get_next_given_traject(self):

        geom  = self.given_geoms[self.istep]
        veloc = self.given_velocities[self.istep]
        
        self.istep += 1

        #print(geom) ## Debug code

        return geom, veloc

    
    def print_xyz(self):

        xyz_file = self.localoutput.traject

        n_atom = len(self.atomparams)

        xyz_file.write("%d\n" % n_atom)

        xyz_file.write( "T= %20.12f fs ( STEP %d ) \n" % (self.t*AU2SEC*1.0e15, self.istep) )

        for i_atom in range(n_atom):

            elem = self.atomparams[i_atom].elem
            coord = self.position[3*i_atom:3*i_atom+3] * AU2ANGST

            xyz_file.write("%s %20.12f %20.12f %20.12f\n" % (
                elem, coord[0], coord[1], coord[2]
            ) )

        return
 

    def print_estate_info(self):
        
        e_coeff_file = self.localoutput.e_coeff
        e_popul_file = self.localoutput.e_popul
        pec_file     = self.localoutput.pec

        n_estate = self.get_n_estate()

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.t*AU2SEC*1.0e15)

        e_coeff_file.write(time_str)
        e_popul_file.write(time_str)
        pec_file.write(time_str)

        e_coeffs = self.e_part.get_e_coeffs()
        estate_energies = self.e_part.get_estate_energies()

        for i_state in range(n_estate):
            
            e_coeff_file.write( " %20.12f+%20.12fj" % ( e_coeffs[i_state].real, e_coeffs[i_state].imag ) )
            e_popul_file.write( " %20.12f" % ( abs(e_coeffs[i_state])**2 ) )
            pec_file.write( " %20.12f" % estate_energies[i_state] )

        e_coeff_file.write("\n")
        e_popul_file.write("\n")
        pec_file.write("\n")

        return


    def print_e_ortho(self):
        
        e_ortho_file = self.localoutput.e_ortho

        n_mo = len(self.e_part.csc)

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.t*AU2SEC*1.0e15)

        e_ortho_file.write(time_str)

        for i_mo in range(n_mo):
            
            e_ortho_file.write( " %20.12f" % self.e_part.csc[i_mo] )

        e_ortho_file.write( " TOTAL NELEC %20.12f\n" % np.sum(self.e_part.csc) )

        return


    def print_thermodynamics(self):
        
        energy_file = self.localoutput.energy

        temperature = self.get_temperature()

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.t*AU2SEC*1.0e15)

        energy_file.write(time_str)

        energy_file.write(
            "%20.12f %20.12f %20.12f %20.12f" % (temperature, self.Ekin, self.Epot, self.EpotGS)
        )

        energy_file.write("\n")

        return


    def print_results(self):
        
        if self.print_xyz_interval != 0 and (self.istep % self.print_xyz_interval) == 0:

            self.print_xyz()

            self.print_estate_info()

            self.print_e_ortho()

            self.print_thermodynamics()

        if self.istep > 0 and (self.istep % self.flush_interval) == 0:

            self.localoutput.flush()

        return
