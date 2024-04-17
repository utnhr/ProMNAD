#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy.linalg as sp
from copy import deepcopy

from constants import H_DIRAC, ANGST2AU, AU2EV, ONEOVER24
import utils
from settingsmodule import load_setting
from integratormodule import Integrator
#from interface_dftbplus import dftbplus_manager
from calcconfig import init_qc_engine
import struct

class Electronic_state:
    
    electronic_state_count = 0
    
    def __init__(self, settings, atomparams, e_coeffs, position, velocity, dt, istep, construct_initial_gs = True):
        
        self.electronic_state_id = Electronic_state.electronic_state_count
        Electronic_state.electronic_state_count += 1

        self.active_occ_mos      = load_setting(settings, 'active_occ_mos')
        self.active_vir_mos      = load_setting(settings, 'active_vir_mos')
        self.qc_program          = load_setting(settings, ('engine', 'type') )

        self.basis               = load_setting(settings, 'basis')
        self.excitation          = load_setting(settings, 'excitation')

        self.reconst_interval    = load_setting(settings, 'reconst_interval')

        self.integmethod         = load_setting(settings, 'integrator')
        
        self.e_coeffs            = e_coeffs # np.array (n_estate)
        self.old_e_coeffs        = deepcopy(self.e_coeffs)
        #### order of linear coefficients ####
        #   case of 'configuration' & 'cis':
        #       

        self.estate_energies     = None
        self.old_estate_energies = None

        self.old_position        = deepcopy(position)
        self.old_velocity        = deepcopy(velocity)
        self.position            = deepcopy(position)
        self.velocity            = deepcopy(velocity)

        self.H                   = None
        self.S                   = None
        self.Sinv                = None
        self.deriv_coupling      = None

        self.old_H               = None
        self.old_S               = None
        self.old_Sinv            = None

        self.dt_deriv            = load_setting(settings, 'dt_deriv')

        self.atomparams          = deepcopy(atomparams)

        self.dt                  = dt

        self.is_open_shell       = False

        self.e_coeffs_tderiv     = np.zeros_like(e_coeffs)
        self.old_e_coeffs_tderiv = np.zeros_like(e_coeffs)

        self.i_step              = istep
        
        self.t_e_coeffs           = 0.0
        self.t_molecular_orbitals = 0.0

        self.is_edyn_initialized  = False

        self.integrator = Integrator()

        if self.qc_program == 'dftb+':
            self.dftbplus_instance = init_qc_engine(settings, "%d" % self.electronic_state_id)
        else:
            utils.stop_with_error("Unknown QC program %s .\n" % self.qc_program)

        if construct_initial_gs:
            self.reconstruct_gs(is_initial = True)
        else:
            self.gs_energy        = None
            self.gs_filling       = None
            self.init_mo_energies = None
            self.mo_coeffs        = None
            self.old_mo_coeffs    = None
            self.n_elec           = None
            self.n_MO             = None
            self.n_AO             = None
            self.gs_rho           = None
            self.initial_gs_energy = None

        return


    def set_next_position(self, position):
        self.next_position = deepcopy(position)
        return


    def update_position(self):
        self.old_position  = deepcopy(self.position)
        self.position      = deepcopy(self.next_position)
        self.next_position = None
        return


    def set_next_velocity(self, velocity):
        self.next_velocity = deepcopy(velocity)
        return


    def update_velocity(self):
        self.old_velocity  = deepcopy(self.velocity)
        self.velocity      = deepcopy(self.next_velocity)
        self.next_velocity = None
        return


    #def set_new_time(self, t):
    #    self.t = t
    #    return


    def is_uptodate(self, istep_last):
        
        if self.get_istep() == istep_last:
            return True
        else:
            return False


    def get_istep(self):
        return self.istep


    def get_n_estate(self):
        return len(self.e_coeffs)


    def get_n_occ(self):
        return self.n_occ


    def get_gs_energy(self):
        return self.gs_energy


    def get_atomparams(self):
        return deepcopy(self.atomparams)


    def get_e_coeffs(self):
        return deepcopy(self.e_coeffs)


    def get_old_e_coeffs(self):
        return deepcopy(self.old_e_coeffs)


    def get_e_coeffs_tderiv(self):
        return deepcopy(self.e_coeffs_tderiv)


    def get_old_e_coeffs_tderiv(self):
        return deepcopy(self.old_e_coeffs_tderiv)


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

        #self.t_mos = self.get_istep()

        return


    def make_mo_tderiv(self, t, mo_coeffs, deriv_coupling, Sinv, without_trivial_phase = True, init_mo_energies = None):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        trivial_phase_factor     = self.get_trivial_phase_factor(init_mo_energies, t, invert = False)
        inv_trivial_phase_factor = self.get_trivial_phase_factor(init_mo_energies, t, invert = True)

        def get_mo_dependent_gs_density_matrix(mo_coeffs):
            
            # mo_coeffs can either phase-containing or phaseless MOs
            
            gs_rho = np.zeros( (n_spin, self.n_AO, self.n_AO), dtype = 'float64' )

            for i_spin in range(n_spin):

                scaled_mo_coeffs = np.zeros_like(mo_coeffs[i_spin,:,:])

                for i_MO in range(self.n_MO):

                    scaled_mo_coeffs[i_MO,:] = self.gs_filling[i_spin][i_MO] * mo_coeffs[i_spin,i_MO,:]

                rho = np.dot( np.transpose(mo_coeffs[i_spin,:,:]).conjugate(), scaled_mo_coeffs )

                gs_rho[i_spin, :, :] = np.real(rho[:, :])
            
            return gs_rho
        
        def get_mo_dependent_hamiltonian(mo_coeffs):
            
            # mo_coeffs can either phase-containing or phaseless MOs

            gs_rho = get_mo_dependent_gs_density_matrix(mo_coeffs)
            
            self.dftbplus_instance.go_to_workdir()
            Htmp = self.dftbplus_instance.worker.return_hamiltonian(gs_rho)
            self.dftbplus_instance.return_from_workdir()

            H = np.zeros_like(Htmp, dtype = 'complex128')

            for i_spin in range(n_spin):
                H[i_spin,:,:] = utils.hermitize(Htmp[i_spin,:,:], is_upper_triangle = True)
            
            return H

        if without_trivial_phase:
            mo = mo_coeffs * trivial_phase_factor
        else:
            mo = mo_coeffs

        H = get_mo_dependent_hamiltonian(mo)

        Heff = np.zeros_like(H)

        mo_tderiv = np.zeros_like(mo)

        for i_spin in range(n_spin):

            Heff[i_spin,:,:] = H[i_spin,:,:] - (0.0+1.0j) * deriv_coupling[:,:]
        
            mo_tderiv[i_spin,:,:] = -(0.0+1.0j) * np.dot(
                np.dot( Sinv.astype('complex128'), Heff[i_spin,:,:] ), mo[i_spin,:,:].transpose()
            ).transpose()

        if without_trivial_phase:

            for i_spin in range(n_spin):
                for i_MO in range(self.n_MO):

                    mo_tderiv[i_spin,i_MO,:] = (0.0+1.0j) * init_mo_energies[i_spin,i_MO] * \
                        mo_coeffs[i_spin,i_MO,:] + inv_trivial_phase_factor[i_spin,i_MO,:] * mo_tderiv[i_spin,i_MO,:]

        return mo_tderiv


    def update_molecular_orbitals(self, is_before_initial = False):
        """Update MO energies and coefficients according to TD-KS equation."""
        utils.printer.write_out('Updating MOs: Started.\n')

        self.old_mo_coeffs = deepcopy(self.mo_coeffs)

        new_mo_coeffs = self.propagate_without_trivial_phase(self.mo_coeffs, self.t_molecular_orbitals, self.dt)

        self.mo_coeffs = deepcopy(new_mo_coeffs)
        
        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        for i_spin in range(n_spin):

            mo_midstep = deepcopy(self.mo_coeffs[i_spin,:,:])

            csc = np.dot( mo_midstep, np.dot( self.S.astype('complex128'), mo_midstep.transpose().conj() ) )
            #print('CSC', csc)
            real_diag = np.real(np.diag(csc))
            print('CSC', real_diag)
            ## End Debug code

            ## Debug code
            #H_MO = np.dot( mo_midstep, np.dot(self.H[i_spin,:,:], mo_midstep.transpose().conj() ) )
            #E_MO = np.real(np.diag(H_MO))
            #print( 'E GAP', (E_MO[9]-E_MO[8])*AU2EV )
            ## End Debug code

        self.t_molecular_orbitals += self.dt

        utils.printer.write_out('Updating MOs: Done.\n')

        return


    #def propagate_with_trivial_phase(self, old_mo, mid_mo, mo_tderiv, dt, is_init_step):
    #
    #    if is_init_step:
    #        factor = 1.0
    #    else:
    #        factor = 2.0

    #    new_mo = old_mo + factor * dt * mo_tderiv

    #    return new_mo

    
    def propagate_without_trivial_phase(self, mo, t, dt):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        inv_trivial_phase_factor = self.get_trivial_phase_factor(self.init_mo_energies, t, invert = True)

        mo_nophase = mo * inv_trivial_phase_factor

        if self.integmethod == 'euler':

            new_mo_nophase = self.integrator.euler(
                self.dt, self.t_molecular_orbitals, mo_nophase, self.make_mo_tderiv,
                self.deriv_coupling, self.Sinv, True, self.init_mo_energies,
            )
        
        elif self.integmethod == 'leapfrog':

            new_mo_nophase = self.integrator.leapfrog(
                self.dt, self.t_molecular_orbitals, mo_nophase, self.make_mo_tderiv,
                self.deriv_coupling, self.Sinv, True, self.init_mo_energies,
            )

        elif self.integmethod == 'adams_bashforth_2':

            new_mo_nophase = self.integrator.adams_bashforth_2(
                self.dt, self.t_molecular_orbitals, mo_nophase, self.make_mo_tderiv,
                self.deriv_coupling, self.Sinv, True, self.init_mo_energies,
            )

        elif self.integmethod == 'adams_bashforth_4':

            new_mo_nophase = self.integrator.adams_bashforth_4(
                self.dt, self.t_molecular_orbitals, mo_nophase, self.make_mo_tderiv,
                self.deriv_coupling, self.Sinv, True, self.init_mo_energies,
            )

        elif self.integmethod == 'adams_moulton_2':

            new_mo_nophase = self.integrator.adams_moulton_2(
                self.dt, self.t_molecular_orbitals, mo_nophase, self.make_mo_tderiv,
                self.deriv_coupling, self.Sinv, True, self.init_mo_energies,
            )

        elif self.integmethod == 'adams_moulton_4':

            new_mo_nophase = self.integrator.adams_moulton_4(
                self.dt, self.t_molecular_orbitals, mo_nophase, self.make_mo_tderiv,
                self.deriv_coupling, self.Sinv, True, self.init_mo_energies,
            )

        else:

            utils.stop_with_error("Unknown integrator type %s .\n" % self.integmethod)

        new_mo = new_mo_nophase * self.get_trivial_phase_factor(self.init_mo_energies, t+dt, invert = False)

        return new_mo

    
    def get_trivial_phase_factor(self, mo_energies, t, invert = False):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        trivial_phase = np.zeros( (n_spin, self.n_MO, self.n_AO), dtype = 'complex128' )
    
        if invert:
            factor = 1.0
        else:
            factor = -1.0
        
        for i_spin in range(n_spin):
            for i_MO in range(self.n_MO):

                trivial_phase[i_spin,i_MO,:] = np.exp( factor * (0.0+1.0j) * mo_energies[i_spin,i_MO] * t)

        return trivial_phase

    
    def get_estate_energies(self, return_1d = True): ## placeholder
        """Get energy of each 'electronic state', which is i->a excitation configuration. Approximate state energy as MO energy difference."""

        #return np.zeros_like(self.e_coeffs)

        utils.printer.write_out('Updating electronic state energies: Started.\n')

        if self.is_open_shell:

            utils.stop_with_error('Currently not compatible with open-shell systems.\n')

        else:
        
            mo_energies = self.mo_levels[0,:]
            mo_coeffs   = self.mo_coeffs[0,:,:]

            n_mo  = len(mo_energies)
            #n_occ = self.n_occ
            #n_vir = n_mo - self.n_occ
            
            state_energies = []

            for i_occ in self.active_occ_mos:
                
                row = []
            
                for i_vir in self.active_vir_mos:
            
                    row.append(self.gs_energy + mo_energies[i_vir] - mo_energies[i_occ])
            
                state_energies.append(row)
            
            self.old_estate_energies = deepcopy(self.estate_energies)

            array_1d = np.array(state_energies).flatten()
            
            self.estate_energies = np.insert(array_1d, 0, self.gs_energy)

            self.estate_energies -= self.initial_gs_energy # shift origin to avoid fast phase oscillation

            utils.printer.write_out('Updating electronic state energies: Done.\n')
            
            return deepcopy(self.estate_energies)


    def update_derivative_coupling(self):

        position_2d = utils.coord_1d_to_2d(self.position)
        velocity_2d = utils.coord_1d_to_2d(self.velocity)

        old_position_2d = position_2d - self.dt_deriv * velocity_2d
        #old_position_2d = utils.coord_1d_to_2d(self.old_position)
        new_position_2d = position_2d + self.dt_deriv * velocity_2d
        #new_position_2d = position_2d + self.dt * velocity_2d
        
        if self.qc_program == 'dftb+':
            
            self.dftbplus_instance.go_to_workdir()
            overlap_twogeom_0 = self.dftbplus_instance.worker.return_overlap_twogeom(position_2d,     position_2d)
            overlap_twogeom_1 = self.dftbplus_instance.worker.return_overlap_twogeom(position_2d, new_position_2d)
            overlap_twogeom_2 = self.dftbplus_instance.worker.return_overlap_twogeom(position_2d, old_position_2d)
            overlap_twogeom_3 = self.dftbplus_instance.worker.return_overlap_twogeom(new_position_2d, position_2d)
            overlap_twogeom_4 = self.dftbplus_instance.worker.return_overlap_twogeom(old_position_2d, position_2d)
            self.dftbplus_instance.return_from_workdir()

            #temp1 = overlap_twogeom_1 - overlap_twogeom_0
            #temp2 = overlap_twogeom_2 - overlap_twogeom_0

            #temp1 = ( temp1 + temp1.transpose() ) * 0.5
            #temp2 = ( temp2 + temp2.transpose() ) * 0.5

            temp1 = np.triu(overlap_twogeom_1) + np.triu(overlap_twogeom_3).transpose()
            temp2 = np.triu(overlap_twogeom_2) + np.triu(overlap_twogeom_4).transpose()

            #temp = np.triu(overlap_twogeom_1 - overlap_twogeom_2)
            #temp = np.triu(overlap_twogeom_3 - overlap_twogeom_0)
            #temp = np.triu(temp1 - temp2)
            #temp = np.tril(temp1 - temp2)
            temp = temp1 - temp2
            #temp = np.triu(temp1 + temp2)
            #temp = np.triu(-temp2)
            #temp = np.zeros_like(temp2) ## Debug code

            overlap_twogeom = temp
            #overlap_twogeom = temp.transpose()

        else:

            utils.stop_with_error("Unknown quantum chemistry program %s .\n" % self.qc_program)

        #self.deriv_coupling = overlap_twogeom / self.dt
        #self.deriv_coupling = overlap_twogeom / (2.0 * self.dt)
        self.deriv_coupling = overlap_twogeom / (2.0 * self.dt_deriv)

        #print('POSITION_2D', position_2d) ## Debug code
        #print('OLD_POSITION_2D', old_position_2d) ## Debug code

        #print('TEMP1', temp1)
        #print('DERIV_COUPLING', self.deriv_coupling)
        
        #print(' WARNING: derivative coupling set to zero') ## Debug code
        #self.deriv_coupling = np.zeros_like(self.deriv_coupling) ## Debug code

        return


    def get_tdnac(self): ## placeholder
        
        n_estate = self.get_n_estate()

        tdnac = np.zeros( (n_estate, n_estate), dtype='complex128' )

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        n_occ = len(self.active_occ_mos)
        n_vir = len(self.active_vir_mos)
        n_act = n_occ + n_vir

        #cis_coeffs     = self.e_coeffs[1:].reshape(n_occ, n_vir)
        #old_cis_coeffs = self.old_e_coeffs[1:].reshape(n_occ, n_vir)
        #cis_coeffs_tderiv = (cis_coeffs - old_cis_coeffs) / self.dt

        if self.basis == 'configuration':

            # AO -> MO

            S_pq = np.zeros_like(self.mo_coeffs)
            S_occ = np.zeros( (n_spin, n_occ, n_occ), dtype = 'complex128' ) # for active occ. MOs
            S_vir = np.zeros( (n_spin, n_vir, n_vir), dtype = 'complex128' ) # for active vir. MOs

            for i_spin in range(n_spin):

                S_pq[i_spin,:,:] = np.dot(
                    np.dot(np.conj(self.old_mo_coeffs[i_spin,:,:]), self.deriv_coupling[:,:].astype('complex128')),
                    self.mo_coeffs[i_spin,:,:].transpose()
                )

                S_occ[i_spin,:,:] = S_pq[i_spin,self.active_occ_mos,:][:,self.active_occ_mos]
                S_vir[i_spin,:,:] = S_pq[i_spin,self.active_vir_mos,:][:,self.active_vir_mos]

            # here, a state is an excitation configuration
            # (currently, assuming closed-shell systems)
            
            if self.is_open_shell:
                utils.stop_with_error('Now, not compatible with open-shell systems.\n')
            
            P = np.array( [ [ (-1.0)**abs(j-i) for j in range(n_occ) ] for i in range(n_occ) ] )
            nac_occ = P * S_occ[i_spin,:,:]
            nac_vir = S_vir[i_spin,:,:]

            for k_estate in range(n_estate):

                # be careful that ground state is included
                
                i_occ_k = (k_estate-1) // n_vir
                i_vir_k = (k_estate-1) % n_vir
                    
                for j_estate in range(k_estate, n_estate):

                    if k_estate * j_estate == 0: # gound-excited (and ground-ground) TDNACs are zero
                        
                        val = 0.0

                    elif k_estate == j_estate: # TDNACs between the same states are zero
                        
                        val = 0.0

                    else:

                        i_occ_j = (j_estate-1) // n_vir
                        i_vir_j = (j_estate-1) % n_vir

                        if i_occ_k == i_occ_j:

                            val = nac_vir[i_vir_k, i_vir_j]

                        elif i_vir_k == i_vir_j:
                            
                            val = nac_occ[i_occ_k, i_occ_j]

                    tdnac[k_estate, j_estate] = val
                    tdnac[j_estate, k_estate] = -val

            #for i_occ in range(n_occ):
            #        tdnac[ 1 + i_occ * n_vir : , 1 + i_occ * n_vir : ] = nac_vir[:, :]

        else:

            utils.stop_with_error("Unknown electronic-state basis %s; TDNAC calculation failed.\n" % self.basis)
        
        return tdnac

    
    def get_force(self):
        """Get nuclear force originating from electronic states."""

        force = np.zeros_like(self.position)

        if not self.is_edyn_initialized:
            print('EDYN INITIALIZATION', self.electronic_state_id) ## Debug code
            self.dftbplus_instance.go_to_workdir()
            self.dftbplus_instance.worker.init_elec_dynamics()
            self.dftbplus_instance.return_from_workdir()
            self.is_edyn_initialized = True

        if self.S is None:
            self.update_matrices(is_before_initial = True)

        S = np.zeros( (1, self.n_AO, self.n_AO), dtype = 'float64' )
        Sinv = np.zeros( (1, self.n_AO, self.n_AO), dtype = 'float64' )
        S[0,:,:] = self.S[:,:]
        Sinv[0,:,:] = self.Sinv[:,:]

        #print('SELF.RHO', self.rho) ## Debug code
        
        rho_real = np.zeros_like(self.rho, dtype = 'float64')
        rho_real[:,:,:] = self.rho[:,:,:]
        
        H = np.zeros_like(self.rho)
        self.dftbplus_instance.go_to_workdir()
        tmp = self.dftbplus_instance.worker.return_hamiltonian(rho_real)
        self.dftbplus_instance.return_from_workdir()

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        for i_spin in range(n_spin):
            H[i_spin,:,:] = utils.hermitize(tmp[i_spin,:,:], is_upper_triangle = True)

        self.dftbplus_instance.go_to_workdir()
        force = self.dftbplus_instance.worker.get_ehrenfest_force(H, self.rho, S, Sinv)
        #force = dftbplus_manager.worker.get_ehrenfest_force(self.H, self.rho, S, Sinv)
        self.dftbplus_instance.return_from_workdir()

        n_atom = len(self.atomparams)

        force = force.reshape(3*n_atom)

        #print('FORCE', force) ## Debug code

        return force
    

    def update_matrices(self, is_before_initial = False):

        utils.printer.write_out('Updating hamiltonian and overlap matrices: Started.\n')

        if self.qc_program == 'dftb+':
            
            #position_2d = utils.coord_1d_to_2d(self.position) * ANGST2AU
            position_2d = utils.coord_1d_to_2d(self.position)

            self.dftbplus_instance.go_to_workdir()

            n_AO = sum( self.dftbplus_instance.worker.get_atom_nr_basis() )

            S = self.dftbplus_instance.worker.return_overlap_twogeom(position_2d, position_2d)

            self.dftbplus_instance.worker.set_geometry(position_2d)

            self.dftbplus_instance.worker.update_coordinate_dependent_stuffs()

            self.dftbplus_instance.return_from_workdir()

            self.update_gs_density_matrix()

            self.update_gs_hamiltonian_and_molevels()
            
            if self.i_step % self.reconst_interval == 0:
                self.reconstruct_gs(energy_only = True)

            n_spin = int(self.is_open_shell) + 1

            if self.S is not None:
                self.old_S = deepcopy(self.S)
            if self.Sinv is not None:
                self.old_Sinv = deepcopy(self.Sinv)

            self.S = utils.symmetrize(S, is_upper_triangle = True)

            self.Sinv = np.linalg.inv(self.S)

            if self.old_S is None:
                self.old_S = deepcopy(self.S)
            if self.old_Sinv is None:
                self.old_Sinv = deepcopy(self.Sinv)
            
            self.construct_density_matrix()

            self.update_derivative_coupling()

            self.update_molecular_orbitals(is_before_initial = is_before_initial)

        else:
            
            utils.stop_with_error("Unknown quantum chemistry program %s .\n" % self.qc_program)

        if not is_before_initial:

            self.i_step += 1

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


    def construct_density_matrix(self):
        
        if self.basis == 'configuration' and self.excitation == 'cis' and not self.is_open_shell:

            self.rho = np.zeros_like(self.gs_rho, dtype = 'complex128')

            # Ground-state contribution

            self.rho += self.gs_rho

            # 1-electron excitation contribution

            n_occ = len(self.active_occ_mos)
            n_vir = len(self.active_vir_mos)
            n_act = n_occ + n_vir

            active_mos = np.zeros( (n_act, self.mo_coeffs.shape[2]), dtype = 'complex128' )

            for i_occ, active_occ_imo in enumerate(self.active_occ_mos):

                active_mos[i_occ,:] = self.mo_coeffs[0,active_occ_imo,:]

            for i_vir, active_vir_imo in enumerate(self.active_vir_mos):

                active_mos[n_occ+i_vir,:] = self.mo_coeffs[0,active_vir_imo,:]

            cis_coeffs = self.e_coeffs[1:].reshape(n_occ, n_vir) # MO basis

            rho_oo_mo = -np.dot( cis_coeffs.conjugate(), cis_coeffs.transpose() )
            rho_vv_mo = np.dot( cis_coeffs.transpose().conjugate(), cis_coeffs )
            rho_ov_mo = self.e_coeffs[0].conjugate() * cis_coeffs
            rho_vo_mo = np.conjugate( rho_ov_mo.transpose() )

            rho_cis = np.zeros( (n_act, n_act), dtype = 'complex128' )

            rho_cis[0    :n_occ, 0    :n_occ] = rho_oo_mo[0:n_occ,0:n_occ]
            rho_cis[n_occ:n_act, n_occ:n_act] = rho_vv_mo[0:n_vir,0:n_vir]
            rho_cis[0    :n_occ, n_occ:n_act] = rho_ov_mo[0:n_occ,0:n_vir]
            rho_cis[n_occ:n_act, 0    :n_occ] = rho_vo_mo[0:n_vir,0:n_occ]

            # MO -> AO

            rho_cis_ao = np.dot( active_mos.transpose().conjugate(), np.dot( rho_cis, active_mos ) )

            self.rho[0,:,:] += rho_cis_ao

            ### Debug code
            #self.rho[:,:,:] = 0.0+0.0j
            #with open('DM1.dat', 'rb') as f:
            #    b = f.read()
            #    u = struct.iter_unpack('<d', b)
            #    count = 0
            #    for ivalue, value in enumerate(u):
            #        #print(value[0])
            #        if ivalue == 0:
            #            continue
            #        elif ivalue % 2 == 1:
            #            tmp1 = value[0]
            #        else:
            #            tmp2 = value[0]
            #            count += 1
            #            iraw = ((ivalue-1)//2) // self.n_AO
            #            icol = ((ivalue-1)//2) % self.n_AO
            #            #print(count,iraw,icol)
            #            self.rho[0,iraw,icol] = tmp1 + 1.0j * tmp2
            ### End Debug code

            print( 'TOTAL NELEC      ', np.trace( np.dot(self.rho[0,:,:], self.S ) ) ) ## Debug code

        else:

            if self.is_open_shell:
                string = 'open shell'
            else:
                string = 'closed shell'

            utils.stop_with_error("1-RDM construction failed; not compatible with %s X %s X %s .\n" % (
                    self.basis, self.excitation, string
                )
            )

        return

    
    def reconstruct_gs(self, is_initial = False, energy_only = False):

        if self.qc_program == 'dftb+':
        
            n_atom = len(self.atomparams)

            coords = self.position.reshape(n_atom, 3)

            self.dftbplus_instance.go_to_workdir()
            
            self.dftbplus_instance.worker.set_geometry(coords)

            self.gs_energy = self.dftbplus_instance.worker.get_energy()

            if energy_only:

                self.dftbplus_instance.return_from_workdir()

                return

            self.init_mo_energies, mo_coeffs_real = self.dftbplus_instance.worker.get_molecular_orbitals(
                open_shell = self.is_open_shell
            )

            self.mo_coeffs     = mo_coeffs_real.astype('complex128')
            #self.old_mo_coeffs = None

            self.gs_filling = self.dftbplus_instance.worker.get_filling(open_shell = self.is_open_shell)

            self.dftbplus_instance.return_from_workdir()

            self.n_elec = np.sum(self.gs_filling)

            self.n_MO = np.size(self.mo_coeffs, 2)
            self.n_AO = self.n_MO

            self.update_gs_density_matrix()
            
            if is_initial:
                self.rho = self.gs_rho.astype('complex128')
                self.initial_gs_energy = self.gs_energy

            #self.construct_initial_molecular_orbitals()

        else:

            utils.stop_with_error("Not compatible with quantum chemistry program %s ." % self.qc_program)

        return


    def update_gs_density_matrix(self):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        self.gs_rho = np.zeros( (n_spin, self.n_AO, self.n_AO), dtype = 'float64' )

        # diagonal matrix whose elements are occupation numbers
        f = np.zeros( (self.n_MO, self.n_MO), dtype = 'complex128' )

        for i_spin in range(n_spin):

            scaled_mo_coeffs = np.zeros_like(self.mo_coeffs[i_spin,:,:])

            for i_MO in range(self.n_MO):

                scaled_mo_coeffs[i_MO,:] = self.gs_filling[i_spin][i_MO] * self.mo_coeffs[i_spin,i_MO,:]

            rho = np.dot( np.transpose(self.mo_coeffs[i_spin,:,:]).conjugate(), scaled_mo_coeffs )

            self.gs_rho[i_spin, :, :] = np.real(rho[:, :])
        
        return
    

    def update_gs_hamiltonian_and_molevels(self):
        
        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        self.dftbplus_instance.go_to_workdir()
        Htmp = self.dftbplus_instance.worker.return_hamiltonian(self.gs_rho)
        self.dftbplus_instance.return_from_workdir()

        self.H = np.zeros_like(Htmp, dtype = 'complex128')
        self.mo_levels = np.zeros( (n_spin, self.n_MO), dtype = 'float64')

        for i_spin in range(n_spin):

            self.H[i_spin,:,:] = utils.hermitize(Htmp[i_spin,:,:], is_upper_triangle = True)
        
            H_MO = np.dot(
                self.mo_coeffs[i_spin,:,:], np.dot(self.H[i_spin,:,:], self.mo_coeffs[i_spin,:,:].transpose().conj())
            )

            self.mo_levels[i_spin,:] = np.real(np.diag(H_MO))
