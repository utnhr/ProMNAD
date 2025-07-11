#!-*- coding: utf-8 -* self.given_geoms[:][0]

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy.linalg as sp
from copy import deepcopy
from time import perf_counter_ns

from constants import H_DIRAC, AMU2AU, AU2ANGST, AU2EV, KB_EV, AU2SEC, SEC2AU
import utils
from interface_dftbplus import dftbplus_manager
from electronmodule import Electronic_state
from settingsmodule import load_setting
from integratormodule import Integrator
from files import GlobalOutputFiles, LocalOutputFiles
#from worldsmodule import World

class Tbf:
    """Class of TBFs."""


    overlap_cutoff = 1.0e-6


    @classmethod
    def calculate_goverlap_from_parameters(cls, pos1, pos2, mom1, mom2, phase1, phase2, n_dof, width):
        
        delta_pos   = pos2 - pos1
        delta_mom   = mom2 - mom1
        delta_phase = phase2 - phase1

        mid_pos   = 0.5 * (pos1 + pos2)
        
        vals = []

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
    def get_gaussian_overlap(cls, tbf1, tbf2):
        """Calculate overlap matrix element (for each degree of freedom) between two gaussian wave packets."""

        pos1 = tbf1.get_position()
        pos2 = tbf2.get_position()

        mom1 = tbf1.get_momentum()
        mom2 = tbf2.get_momentum()

        phase1 = tbf1.get_phase()
        phase2 = tbf2.get_phase()

        n_dof = tbf1.get_n_dof()
        width = tbf1.get_width()

        return cls.calculate_goverlap_from_parameters(pos1, pos2, mom1, mom2, phase1, phase2, n_dof, width)


    @classmethod
    def get_gaussian_overlap_delay(cls, tbf_old, tbf_new):
        
        pos1 = tbf_old.get_old_position()
        pos2 = tbf_new.get_position()

        mom1 = tbf_old.get_old_momentum()
        mom2 = tbf_new.get_momentum()

        phase1 = tbf_old.get_old_phase()
        phase2 = tbf_new.get_phase()

        n_dof = tbf_old.get_n_dof()
        width = tbf_old.get_width()
        
        return cls.calculate_goverlap_from_parameters(pos1, pos2, mom1, mom2, phase1, phase2, n_dof, width)

    
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
            np.conj(tbf1.e_part.get_e_coeffs()), tbf2.e_part.get_e_coeffs()
        )


    @classmethod
    def get_wf_overlap_delay(cls, tbf_old, tbf_new, gaussian_overlap_delay):
        """Calculate overlap matrix between two TBFs at different time."""
        s_gw = gaussian_overlap_delay

        e_overlap = 0.0+0.0j

        # overlap for the electronic part
        # note that electronic wavefunctions at t and t+Dt are non-orthogonal

        old_e_coeffs = tbf_old.e_part.get_old_e_coeffs()
        new_e_coeffs = tbf_new.e_part.get_e_coeffs()

        n_estate = new_e_coeffs.size()

        tbf_new.e_part.set_mo_overlap_with_old_other_e_part(tbf_old.e_part)

        for i_estate in range(n_estate):
            for j_estate in range(n_estate):

                a1a2 = np.conj(old_e_coeffs[i_estate]) * new_e_coeffs[j_estate]

                if abs(a1a2) < cls.overlap_cutoff:
                    continue

                e_overlap += tbf_new.e_part.get_overlap_with_old_other_e_part(i_estate, j_estate)

                #e_overlap += a1a2 ## Debug code

        return s_gw.prod() * e_overlap

    
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

        #kin = - 0.5 * H_DIRAC**2 * (
        #    (1j/H_DIRAC) * 2.0 * width * del_R * mid_P - \
        #    width + width**2 * del_R**2 - mid_P**2 / H_DIRAC**2
        #) * s_gw / mass 
        
        in_sigma = - 0.5 * H_DIRAC**2 * (
            (1j/H_DIRAC) * 2.0 * width* del_R * mid_P - \
            width + width * del_R * width * del_R - mid_P * mid_P / H_DIRAC**2 # ?????
        ) / mass 

        kin = np.sum(in_sigma) * s_gw.prod()

        #kin = - 0.5 * H_DIRAC**2 * (
        #    (1j/H_DIRAC) * 2.0 * np.dot(width*del_R,mid_P) - \
        #    np.sum(width) + np.dot(width*del_R,width*del_R) - np.dot(mid_P,mid_P) / H_DIRAC**2 # ?????
        #) * s_gw.prod() / mass 

        if each_degree:
            return kin
        else:
            #return kin.prod()
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
        
        ### Debug code
        #print('GAUSSIAN OVERLAP', gaussian_overlap.prod())
        #print('KINE', kinE)
        #print('POTENTIALS', potentials)
        #print('TDNACS', tdnacs)
        ### End Debug code

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

        e_coeffs_tauderiv_2 = tbf2.e_part.get_e_coeffs_tauderiv()

        term1 = gwp_derivative * np.dot( np.conjugate(e_coeffs_1), e_coeffs_2)
        term2 = gaussian_overlap.prod() * np.dot( np.conjugate(e_coeffs_1), e_coeffs_tauderiv_2 )
        
        return term1 + term2


    def __init__(
            self, settings, atomparams, position, n_dof, n_estate, tbf_id,
            momentum=None, mass=None, width=None, phase=None,
            e_coeffs_nophase=None, e_coeffs_e_int=None, initial_gs_energy=None,
            istep=0, is_fixed=False,
            e_part_matrices = None,
        ):
        
        # Global settings

        self.settings = settings

        # Load necessary settings

        self.dtau = load_setting(settings, 'dtau')
        self.read_traject = load_setting(settings, 'read_traject')
        self.print_xyz_interval = load_setting(settings, 'print_xyz_interval')
        self.dump_mo_tdnac_interval = load_setting(settings, 'print_mo_tdnac_interval')
        self.dump_mo_coeffs_interval = load_setting(settings, 'print_mo_coeffs_interval')
        self.flush_interval = load_setting(settings, 'flush_interval')
        self.integmethod = load_setting(settings, 'integrator')
        self.alpha = load_setting(settings, 'alpha')
        self.e_coeffs_are_trivial = load_setting(settings, 'e_coeffs_are_trivial')
        self.calc_nonorthogonality_interval = load_setting(settings, 'calc_nonorthogonality_interval')

        # Time-independent values

        self.atomparams    = atomparams        # dictionary { 'elems': array, 'angmom_table': array (optional, for DFTB+) }
        self.n_dof         = n_dof             # integer
        self.n_estate      = n_estate          # integer
        self.init_position = position          # np.array (n_dof)
        if momentum is not None:
            self.init_momentum = momentum # np.array (n_dof)
        else:
            self.init_momentum = np.zeros(n_dof, dtype='float64')
        self.mass          = mass              # np.array (n_dof)
        self.width         = width             # np.array (n_dof)
        self.init_istep    = istep             # integer, index of step
        self.tbf_id        = tbf_id            # integer (start 0)
        self.is_fixed      = is_fixed          # logical
        #self.world_id      = world_id          # int (start 0)
        self.dt            = self.alpha * self.dtau # for position and momentum integration
        
        # Initial time
        self.istep = self.init_istep # Position and momentum steps
        t = self.istep * self.dt

        # Time-dependent values and "clocks"
        # "old" means "at the previous nuclear position"

        if phase is not None:
            self.phase = phase # np.array (n_dof)
        else:
            self.phase = np.zeros(n_dof, dtype='float64')
        self.old_phase = deepcopy(self.phase)
        self.t_phase = t

        #self.gs_energy     = initial_gs_energy # None or float
        #self.t_gs_energy   = t

        self.position       = deepcopy(self.init_position)
        self.old_position   = deepcopy(self.position)
        self.t_position     = t

        self.momentum       = deepcopy(self.init_momentum)
        self.old_momentum   = deepcopy(self.momentum)
        self.e_coeffs_are_trivial = load_setting(settings, 'e_coeffs_are_trivial')
        self.t_momentum     = t

        if self.read_traject:

            given_geoms       = load_setting(settings, 'given_geoms')
            given_velocities  = load_setting(settings, 'given_velocities')
            given_time_frames = load_setting(settings, 'given_time_frames')

            self.given_geoms       = deepcopy(given_geoms)
            self.given_velocities  = deepcopy(given_velocities)
            self.given_time_frames = deepcopy(given_time_frames)

            position, velocity = self.get_next_given_traject(t)

            self.position = position
            self.momentum = self.mass * velocity
            self.old_position = deepcopy(self.position)
            self.old_momentum = deepcopy(self.momentum)

            self.istep = self.init_istep # istep has been updated in get_next_given_traject(), but is should be set to 0

        self.force        = np.zeros_like(self.position)
        self.old_force    = np.zeros_like(self.position)
        self.t_force      = t

        self.gs_force     = np.zeros_like(self.position)
        self.old_gs_force = np.zeros_like(self.position)
        self.t_gs_force   = t

        #self.e_dot = np.zeros(n_estate) # dc/dt

        #self.tbf_id = world.total_tbf_count + 1

        if e_coeffs_nophase is not None:
            self.e_coeffs_nophase = deepcopy(e_coeffs_nophase)
            self.e_coeffs_e_int   = deepcopy(e_coeffs_e_int)
            e_coeffs = self.e_coeffs_nophase * np.exp(
                (-1.0j/H_DIRAC) * self.e_coeffs_e_int
            )
        self.t_e_coeffs_nophase  = t
        self.t_e_coeffs          = t
        self.t_e_coeffs_e_int    = t
        self.t_e_coeffs_tauderiv = t

        self.tau_e_coeffs_nophase  = self.alpha * t
        self.tau_e_coeffs          = self.alpha * t
        self.tau_e_coeffs_e_int    = self.alpha * t
        self.tau_e_coeffs_tauderiv = self.alpha * t

        self.Epot = 0.0
        self.t_Epot = t

        self.EpotGS = 0.0
        self.t_EpotGS = t

        self.dEpotGS = 0.0
        self.t_dEpotGS = t

        self.Ekin = 0.0
        self.t_Ekin = t

        #self.t       = self.dt * self.istep
        
        # Electronic part instance

        # check that every clocks are consistent

        self.e_part = Electronic_state(
            settings, atomparams,
            e_coeffs, self.t_e_coeffs,
            self.get_position(), self.t_position,
            self.get_velocity(), self.t_momentum,
            self.dtau, self.alpha, self.istep,
            matrices = e_part_matrices,
        )

        # Integrators

        #self.e_coeffs_nophase_integrator = Integrator(self.integmethod, mode = 'chasing')
        #self.e_coeffs_e_int_integrator   = Integrator(self.integmethod, mode = 'chasing')
        #self.phase_integrator            = Integrator(self.integmethod, mode = 'chasing')
        self.e_coeffs_nophase_integrator = Integrator('adams_moulton_2', mode = 'chasing') ## Debug code
        self.e_coeffs_e_int_integrator   = Integrator('adams_moulton_2', mode = 'chasing') ## Debug code
        self.phase_integrator            = Integrator('adams_moulton_2', mode = 'chasing') ## Debug code
        
        # Status

        self.is_alive = True
        self.childs = []
        
        # I/O
        self.localoutput   = LocalOutputFiles(self.tbf_id)

        # evaluate closeness of occ/vir Hilbert spaces to those of canonical orbitals
        if self.istep % self.calc_nonorthogonality_interval == 0:
            self.det_S_occ, self.det_S_vir = self.e_part.non_orthogonality()

        # Print initial step
        self.print_results()

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


    def spawn(self, i_state, tbf_id):

        print('I_STATE', i_state) ## Debug code

        baby_e_coeffs_nophase = deepcopy(self.e_coeffs_nophase)
        baby_e_coeffs_nophase[:] = 0.0+0.0j
        baby_e_coeffs_nophase[i_state] = 1.0+0.0j

        baby_e_coeffs_e_int = deepcopy(self.e_coeffs_e_int)
        #baby_e_coeffs_e_int[:] = 0.0+0.0j
        #baby_e_coeffs_e_int[i_state] = self.e_coeffs_e_int[i_state]
        baby_e_coeffs_e_int[:] = self.e_coeffs_e_int[:]

        baby_e_part_matrices = self.e_part.dump_matrices()

        baby = Tbf(
            settings         = deepcopy(self.settings),
            atomparams       = deepcopy(self.atomparams),
            position         = deepcopy(self.position),
            momentum         = deepcopy(self.momentum),
            width            = deepcopy(self.width),
            istep            = self.istep,
            n_dof            = self.n_dof,
            n_estate         = self.n_estate,
            e_coeffs_are_trivial = self.e_coeffs_are_trivial,
            e_coeffs_nophase = baby_e_coeffs_nophase,
            e_coeffs_e_int   = baby_e_coeffs_e_int,
            mass             = deepcopy(self.mass),
            tbf_id           = tbf_id,
            e_part_matrices  = baby_e_part_matrices,
        )
        
        #instance_vars = vars(self)

        ## |c_i|^2 = 1 and c_j = 0 (j =/= i) for baby TBF

        #c_i = self.e_part.e_coeffs[i_state]

        #baby.e_part.e_coeffs[:] = 0.0+0.0j
        #baby.e_part.e_coeffs[i_state] = c_i

        # c_i = 0 and c_j (j =/= i) rescaled for parent TBF

        self.e_coeffs_nophase[i_state] = 0.0+0.0j
        new_norm = np.linalg.norm(self.e_coeffs_nophase)
        print('NEW_NORM', new_norm) ## Debug code
        self.e_coeffs_nophase /= new_norm

        self.e_part.e_coeffs[i_state] = 0.0+0.0j
        self.e_part.e_coeffs[:] /= new_norm
        self.e_part.e_coeffs_tderiv[i_state] = 0.0+0.0j
        self.e_part.e_coeffs_tderiv[:] /= new_norm

        # reset integrators for discontinuous change in e_coeffs

        baby.e_coeffs_nophase_integrator.reset()
        self.e_coeffs_nophase_integrator.reset()
        baby.e_coeffs_e_int_integrator.reset()
        self.e_coeffs_e_int_integrator.reset()

        # register as a child TBF

        self.childs.append(baby)
        
        return baby


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
        self.force, self.t_force = self.e_part.get_force(get_time = True)

        self.old_gs_force = self.gs_force
        self.gs_force, self.t_gs_force = self.e_part.get_force(gs_force = True, get_time = True)
        
        ### Debug code
        #force = self.e_part.pyscf_instance.ks.Gradients().kernel()
        #n_atom = len(self.atomparams)
        #print(self.gs_force + force.reshape(n_atom*3))
        ### End Debug code

        return


    def get_force(self):
        return deepcopy(self.force)


    def set_new_istep(self, istep, e_part_too=False):
        
        self.istep = istep

        if e_part_too:
            self.e_part.set_new_istep(self.istep)

        return


    def make_e_coeffs_nophase_tderiv(self, t, e_coeffs_nophase, H_el_ndiag):

        e_coeffs_nophase_tderiv = (-1.0j / H_DIRAC) * np.dot(
            H_el_ndiag, e_coeffs_nophase
        )

        return e_coeffs_nophase_tderiv


    def make_e_coeffs_e_int_tderiv(self, t, e_coeffs_e_int, H_el_diag):

        return deepcopy(H_el_diag) # 1d array of state energies


    def initialize_e_coeffs_integrator(self):

        tdnac = self.e_part.get_tdnac()

        estate_energies = self.e_part.get_estate_energies()
        
        if self.e_part.initial_estate_energies is None:
            self.e_part.initial_estate_energies = [ estate_energies[0] for i_state in range( len(estate_energies) ) ]

        estate_energies -= self.e_part.initial_estate_energies
        
        H_el_ndiag = -1.0j * H_DIRAC * tdnac
        H_el_diag  = estate_energies

        self.e_coeffs_nophase_integrator.initialize_history(
            self.t_e_coeffs_nophase, self.e_coeffs_nophase,
            self.make_e_coeffs_nophase_tderiv, H_el_ndiag,
        )

        self.e_coeffs_e_int_integrator.initialize_history(
            self.t_e_coeffs_e_int, self.e_coeffs_e_int,
            self.make_e_coeffs_e_int_tderiv, H_el_diag,
        )

        return


    def update_electronic_part(self, dtau, alpha=1.0):

        # (in the initial step) initialize integrators
        if not self.e_coeffs_nophase_integrator.is_history_initialized:
            self.initialize_e_coeffs_integrator()

        # update position and velocity for electronic part

        self.e_part.set_next_position( self.get_position(), self.t_position )
        self.e_part.set_next_velocity( self.get_velocity(), self.t_momentum )
        #self.e_part.set_next_istep( self.get_istep() )

        self.e_part.update_position()
        self.e_part.update_velocity()

        # update electronic wavefunc and relevant physical quantities

        self.e_part.update_position_dependent_quantities()

        # propagate MOs
        
        ts = perf_counter_ns()
        self.e_part.propagate_molecular_orbitals()
        te = perf_counter_ns()
        print("Time for MO propagation: %12.3f sec.\n" % ((te-ts)/1.0e+9)) ## Debug code

        # update MO-dependent quantities

        self.e_part.update_mo_dependent_quantities()

        estate_energies = self.e_part.get_estate_energies()
        
        # origin of electronic state energies

        if self.e_part.initial_estate_energies is None:
            self.e_part.initial_estate_energies = [ estate_energies[0] for i_state in range( len(estate_energies) ) ]

        estate_energies -= self.e_part.initial_estate_energies

        # construct electronic Hamiltonian
        # and update time derivative of electronic coeffs

        #self.e_part.update_tdnac()

        tdnac = self.e_part.get_tdnac()
        
        n_estate = self.e_part.get_n_estate()

        H_el_ndiag = -1.0j * H_DIRAC * tdnac
        H_el_diag  = estate_energies
        
        dt = alpha * dtau
        t = dt * self.get_istep()

        # integrate electronic state coeffcients

        if n_estate == 1 and self.e_coeffs_are_trivial:

            self.e_coeffs = np.array( [ 1.0+0.0j ] )
            self.e_coeffs_nophase = deepcopy(self.e_coeffs)

        else:

            self.e_coeffs_nophase = self.e_coeffs_nophase_integrator.engine(
                dtau, self.tau_e_coeffs_nophase, self.e_coeffs_nophase,
                self.make_e_coeffs_nophase_tderiv, H_el_ndiag,
            )
            self.t_e_coeffs_nophase += dt
            self.tau_e_coeffs_nophase += dtau

            self.e_coeffs_e_int = self.e_coeffs_e_int_integrator.engine(
                dtau, self.tau_e_coeffs_e_int, self.e_coeffs_e_int,
                self.make_e_coeffs_e_int_tderiv, H_el_diag,
            )
            self.t_e_coeffs_e_int += dt
            self.tau_e_coeffs_e_int += dtau

            e_coeffs = self.e_coeffs_nophase * np.exp(
                ( (-1.0j)/H_DIRAC ) * self.e_coeffs_e_int
            )

        self.t_e_coeffs += dt
        self.tau_e_coeffs += dtau

        self.e_part.set_new_e_coeffs(e_coeffs, self.t_e_coeffs)

        # here e_coeffs_tderiv is not directly used for integration of e_coeffs;
        # just for reuse for update of TBF coefficients

        #term1 = self.make_e_coeffs_nophase_tderiv(
        #    self.t_e_coeffs_nophase, self.e_coeffs_nophase, H_el_ndiag,
        #) * np.exp( (-1.0j/H_DIRAC) * self.e_coeffs_e_int )
        #term2 = self.e_coeffs_nophase * (-1.0j/H_DIRAC) * estate_energies

        #e_coeffs_tderiv = term1 + term2

        H_el = H_el_ndiag + np.diag(estate_energies)
        e_coeffs_tauderiv = (-1.0j / H_DIRAC) * np.dot(
            H_el, e_coeffs
        )
        self.t_e_coeffs_tderiv = self.t_e_coeffs
        self.tau_e_coeffs_tderiv = self.tau_e_coeffs

        self.e_part.set_new_e_coeffs_tauderiv(e_coeffs_tauderiv, self.t_e_coeffs_tauderiv)

        # update e_coeffs_dependent stuffs

        self.e_part.update_e_coeffs_dependent_quantities()

        self.set_new_istep( self.get_istep() + 1 )

        # evaluate closeness of occ/vir Hilbert spaces to those of canonical orbitals
        if self.istep % self.calc_nonorthogonality_interval == 0:
            self.det_S_occ, self.det_S_vir = self.e_part.non_orthogonality()

        # sum up results of current time step

        self.print_results()

        return


    def dgdt(self, t, phase): # dgamma/dt; time-dependence of phase

        momentum = self.get_momentum()
        velocity = self.get_velocity()
        
        return 0.5 * np.dot(momentum, velocity)


    def update_position_and_velocity(self, dt):

        if not self.phase_integrator.is_history_initialized:
            self.phase_integrator.initialize_history(self.t_phase, self.phase, self.dgdt)
        
        old_position = self.get_old_position()
        velocity     = self.get_velocity()

        if self.read_traject:
            position, velocity = self.get_next_given_traject(self.t_position)
        else:
            #position = old_position + 2.0 * velocity * dt
            position = self.get_position() + velocity * dt + 0.5 * ( self.force / self.get_mass_au() ) * dt**2

        old_momentum = self.get_old_momentum()
        
        if not self.read_traject:
            self.update_force()
            self.t_force += dt

        if self.read_traject:
            momentum = velocity * self.get_mass_au()
        else:
            #momentum = old_momentum + 2.0 * force * dt
            momentum = self.get_momentum() + 0.5 * (self.old_force + self.force) * dt
        
        self.old_phase = deepcopy(self.phase)
        self.phase = self.phase_integrator.engine(dt, self.t_phase, self.phase, self.dgdt)
        self.t_phase += dt

        if self.is_fixed:
            position = old_position
            momentum = old_momentum

        delta = 0.5 * (position - old_position)

        self.Epot   -= np.dot(delta, self.force)
        self.t_Epot += dt

        self.dEpotGS = -np.dot(delta, self.gs_force)
        self.t_dEpotGS += dt

        self.Ekin    = sum( 0.5 * momentum**2 / self.get_mass_au() )
        self.t_Ekin += dt
        
        self.EpotGS += self.dEpotGS
        self.t_EpotGS += dt
        self.e_part.modify_gs_energy(self.dEpotGS, self.t_EpotGS)

        self.set_new_position(position)
        self.t_position += dt

        self.set_new_momentum(momentum)
        self.t_momentum += dt

        return


    def get_temperature(self):
        
        e_kin_per_dof_ev = AU2EV * self.Ekin / np.size(self.position)
        temp_kelvin = e_kin_per_dof_ev / (0.5 * KB_EV)
        return temp_kelvin

    
    def get_next_given_traject(self, t):

        i_pre, i_nex = utils.binary_search(self.given_time_frames, t)

        if i_pre == i_nex:
            
            x = 1.0

        else:

            t_pre = self.given_time_frames[i_pre]
            t_nex = self.given_time_frames[i_nex]

            x = (t - t_pre) / (t_nex - t_pre)

        geom  = self.given_geoms[i_pre] * (1.0-x) + self.given_geoms[i_nex] * x
        veloc = self.given_velocities[i_pre] * (1.0-x) + self.given_velocities[i_nex] * x
        
        return geom, veloc

    
    def print_xyz(self):

        xyz_file = self.localoutput.traject

        n_atom = len(self.atomparams)

        xyz_file.write("%d\n" % n_atom)

        xyz_file.write( "T= %20.12f fs ( STEP %d ) \n" % (self.t_position*AU2SEC*1.0e15, self.istep) )

        for i_atom in range(n_atom):

            elem = self.atomparams[i_atom].elem
            coord = self.position[3*i_atom:3*i_atom+3] * AU2ANGST

            xyz_file.write("%s %20.12f %20.12f %20.12f\n" % (
                elem, coord[0], coord[1], coord[2]
            ) )

        return


    def print_veloc(self):
        
        xyz_file = self.localoutput.velocity

        n_atom = len(self.atomparams)

        xyz_file.write("%d\n" % n_atom)

        xyz_file.write( "T= %20.12f fs ( STEP %d ) \n" % (self.t_position*AU2SEC*1.0e15, self.istep) )

        veloc = ( self.momentum / self.get_mass_au() ) * AU2ANGST * SEC2AU * 1.0e-15 # a.u. -> Angst/fs

        for i_atom in range(n_atom):

            elem = self.atomparams[i_atom].elem
            coord = veloc[3*i_atom:3*i_atom+3]

            xyz_file.write("%s %20.12f %20.12f %20.12f\n" % (
                elem, coord[0], coord[1], coord[2]
            ) )

        return
 

    def print_estate_info(self):
        
        e_coeff_file = self.localoutput.e_coeff
        e_popul_file = self.localoutput.e_popul
        pec_file     = self.localoutput.pec

        n_estate = self.get_n_estate()

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.t_e_coeffs*AU2SEC*1.0e15)

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
        e_popul_file.write( " TOTAL %20.12f \n" % np.linalg.norm(e_coeffs) )
        pec_file.write("\n")

        return


    def print_e_ortho(self):
        
        e_ortho_file = self.localoutput.e_ortho

        n_mo = len(self.e_part.csc)

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.e_part.t_csc*AU2SEC*1.0e15)

        e_ortho_file.write(time_str)

        for i_mo in range(n_mo):
            
            e_ortho_file.write( " %20.12f" % self.e_part.csc[i_mo] )

        e_ortho_file.write( " TOTAL NELEC %20.12f\n" % np.sum(self.e_part.csc) )

        return

    
    def print_mo_level(self):

        if self.e_part.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        mo_level_file = self.localoutput.mo_level

        n_mo = len(self.e_part.csc)

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.e_part.t_mo_levels*AU2SEC*1.0e15)

        mo_level_file.write(time_str)

        for i_spin in range(n_spin):

            for i_mo in range(n_mo):
            
                mo_level_file.write( " %20.12f" % self.e_part.mo_levels[i_spin,i_mo] )

        mo_level_file.write( "\n" % np.sum(self.e_part.csc) )

        return


    def print_thermodynamics(self):
        
        energy_file = self.localoutput.energy

        temperature = self.get_temperature()

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.t_Ekin*AU2SEC*1.0e15)

        energy_file.write(time_str)

        energy_file.write(
            "%20.12f %20.12f %20.12f %20.12f" % (temperature, self.Ekin, self.Epot, self.EpotGS)
        )

        energy_file.write("\n")

        return


    def print_det(self):

        det_file = self.localoutput.det

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.e_part.t_csc*AU2SEC*1.0e15)

        det_file.write(time_str)

        det_file.write( " %20.12f+%20.12fj,%20.12f,%20.12f+%20.12fj,%20.12f\n" % (
            self.det_S_occ.real, self.det_S_occ.imag, abs(self.det_S_occ),
            self.det_S_vir.real, self.det_S_vir.imag, abs(self.det_S_vir),
        ) )

        return

    
    def print_mo_level(self):

        if self.e_part.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        mo_level_file = self.localoutput.mo_level

        n_mo = len(self.e_part.csc)

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.e_part.t_mo_levels*AU2SEC*1.0e15)

        mo_level_file.write(time_str)

        for i_spin in range(n_spin):

            for i_mo in range(n_mo):
            
                mo_level_file.write( " %20.12f" % self.e_part.mo_levels[i_spin,i_mo] )

        mo_level_file.write( "\n" % np.sum(self.e_part.csc) )

        return


    def print_cmo_level(self):
        
        if self.e_part.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        cmo_level_file = self.localoutput.cmo_level

        n_mo = len(self.e_part.csc)

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.e_part.t_cmo*AU2SEC*1.0e15)

        cmo_level_file.write(time_str)

        for i_spin in range(n_spin):

            for i_mo in range(n_mo):
            
                cmo_level_file.write( " %20.12f" % self.e_part.cmo_levels[i_spin,i_mo] )

        cmo_level_file.write( "\n" % np.sum(self.e_part.csc) )

        return


    def print_thermodynamics(self):
        
        energy_file = self.localoutput.energy

        temperature = self.get_temperature()

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.t_Ekin*AU2SEC*1.0e15)

        energy_file.write(time_str)

        energy_file.write(
            "%20.12f %20.12f %20.12f %20.12f" % (temperature, self.Ekin, self.Epot, self.EpotGS)
        )

        energy_file.write("\n")

        return


    def print_mo_coeffs(self):

        if self.e_part.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        mo_coeffs_file = self.localoutput.mo_coeffs

        time_str = "STEP %12d T= %20.12f fs\n" % (self.istep, self.e_part.t_mo_coeffs_nophase*AU2SEC*1.0e15)

        mo_coeffs_file.write(time_str)

        for i_spin in range(n_spin):

            for i_mo in range(self.e_part.n_MO):

                for i_ao in range(self.e_part.n_AO):

                    coeff = self.e_part.mo_coeffs[i_spin,i_mo,i_ao]

                    mo_coeffs_file.write( "%20.12f+%20.12fj," % (coeff.real, coeff.imag) )

                mo_coeffs_file.write("\n")

        return


    def print_cmo_coeffs(self):
        
        if self.e_part.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        cmo_coeffs_file = self.localoutput.cmo_coeffs

        time_str = "STEP %12d T= %20.12f fs\n" % (self.istep, self.e_part.t_mo_coeffs_nophase*AU2SEC*1.0e15)

        cmo_coeffs_file.write(time_str)

        for i_spin in range(n_spin):

            for i_mo in range(self.e_part.n_MO):

                for i_ao in range(self.e_part.n_AO):

                    coeff = self.e_part.cmo_coeffs[i_spin,i_mo,i_ao]

                    cmo_coeffs_file.write("%20.12f," % coeff)

                cmo_coeffs_file.write("\n")

        return


    def print_mo_tdnac(self):

        if self.e_part.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        mo_tdnac_file = self.localoutput.mo_tdnac

        time_str = "STEP %12d T= %20.12f fs\n" % (self.istep, self.e_part.t_mo_coeffs_nophase*AU2SEC*1.0e15)

        mo_tdnac_file.write(time_str)

        max_tdnac_norm = 0.0
        max_tdnac_mopair = None

        for i_spin in range(n_spin):

            for i_mo in range(self.e_part.n_MO):

                for j_mo in range(self.e_part.n_MO):

                    tdnac = self.e_part.mo_tdnac[i_spin,i_mo,j_mo]

                    if i_mo == j_mo:

                        tdnac -= -(0.0+1.0j) * self.e_part.mo_levels[i_spin,i_mo]

                    tdnac /= self.e_part.alpha # d/dtau -> d/dt
                    
                    norm = abs(tdnac)

                    if norm > max_tdnac_norm:

                        max_tdnac_norm = norm
                        max_tdnac_mopair = (i_mo, j_mo)

                    mo_tdnac_file.write( "%20.12f+%20.12fj," % (tdnac.real, tdnac.imag) )

                mo_tdnac_file.write("\n")

            utils.Printer.write_out(
                "Max. MO TDNAC: %20.12f (MO %d and %d)\n" % (
                    max_tdnac_norm, max_tdnac_mopair[0], max_tdnac_mopair[1]
                )
            )

        return


    def print_phase_cancellation_angles(self):

        if self.e_part.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        pc_angle_file = self.localoutput.pc_angle

        n_mo = len(self.e_part.csc)

        time_str = "STEP %12d T= %20.12f fs" % (self.istep, self.e_part.t_mo_levels*AU2SEC*1.0e15)

        pc_angle_file.write(time_str)

        for i_spin in range(n_spin):

            for i_mo in range(n_mo):

                angle, res = self.e_part.get_phase_cancellation_angle(i_spin, i_mo)
            
                pc_angle_file.write( " %20.14f:%20.14f," % (angle, res)  )

        pc_angle_file.write("\n")

        return


    def print_results(self):
        
        if self.print_xyz_interval != 0 and (self.istep % self.print_xyz_interval) == 0:

            self.print_xyz()

            self.print_veloc()

            self.print_estate_info()

            self.print_e_ortho()

            self.print_mo_level()

            self.print_thermodynamics()

        if self.istep % self.calc_nonorthogonality_interval == 0:

            self.print_det()
            self.print_cmo_level()
            self.print_cmo_coeffs()

        if self.istep % self.dump_mo_tdnac_interval == 0:
            
            self.print_mo_tdnac()

        if self.istep % self.dump_mo_coeffs_interval == 0:
            
            self.print_mo_coeffs()
            self.print_phase_cancellation_angles()

        if self.istep > 0 and (self.istep % self.flush_interval) == 0:

            self.localoutput.flush()

        return
