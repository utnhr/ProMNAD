#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy.linalg as sp
from copy import deepcopy

from constants import H_DIRAC, ANGST2AU
import utils
from settingsmodule import load_setting

from interface_dftbplus import dftbplus_manager

class Electronic_state:
    

    def __init__(self, settings, atomparams, e_coeffs, position, velocity, dt, t, construct_initial_gs = True):

        self.active_occ_mos      = load_setting(settings, 'active_occ_mos')
        self.active_vir_mos      = load_setting(settings, 'active_vir_mos')
        self.qc_program          = load_setting(settings, ('engine', 'type') )

        self.basis               = load_setting(settings, 'basis')
        self.excitation          = load_setting(settings, 'excitation')
        
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

        self.dt_deriv            = load_setting(settings, 'dt_deriv')

        self.atomparams          = deepcopy(atomparams)

        self.dt                  = dt

        self.is_open_shell       = False

        self.e_coeffs_tderiv     = np.zeros_like(e_coeffs)
        self.old_e_coeffs_tderiv = np.zeros_like(e_coeffs)

        self.t                    = t
        self.next_t               = None
        
        self.t_e_coeffs           = self.t
        self.t_molecular_orbitals = self.t
        self.t_estate_energies    = sys.maxsize
        self.t_matrices           = sys.maxsize
        self.t_force              = sys.maxsize
        self.t_tdnac              = sys.maxsize
        self.t_state_coeffs       = sys.maxsize

        self.is_edyn_initialized  = False

        if construct_initial_gs:
            self.construct_initial_gs()
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

        return


    #def set_new_position(self, position):
    #    self.old_position = deepcopy(self.position)
    #    self.position = deepcopy(position)
    #    return


    def set_next_position(self, position):
        self.next_position = deepcopy(position)
        return


    def update_position(self):
        self.old_position  = deepcopy(self.position)
        self.position      = deepcopy(self.next_position)
        self.next_position = None
        return


    #def set_new_velocity(self, velocity):
    #    self.old_velocity = deepcopy(self.velocity)
    #    self.velocity = deepcopy(velocity)
    #    return


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


    def set_next_time(self, t):
        self.next_t = t
        return


    def update_time(self):
        self.t = self.next_t
        return


    def is_uptodate(self, time_last):
        
        #if utils.is_equal_scalar(self.get_t(), time_last):
        if self.get_t() == time_last:
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

        is_initial_step = self.old_mo_coeffs is None

        if is_initial_step:
            self.old_mo_coeffs = np.zeros_like(self.mo_coeffs)

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        for i_spin in range(n_spin):

            mo_midstep = deepcopy(self.mo_coeffs[i_spin,:,:])

            #Heff = self.H[i_spin,:,:] - (0.0+1.0j) * self.deriv_coupling[:,:]
            print(" ##### WARNING: Heff is not correct (for debug) ##### ") ## Debug code
            Heff = - (0.0+1.0j) * self.deriv_coupling[:,:] ## Debug code

            mo_tderiv = -(0.0+1.0j) * np.dot(
                np.dot( self.Sinv.astype('complex128'), Heff ), mo_midstep.transpose()
            ).transpose()

            if is_initial_step:
                self.old_mo_coeffs[i_spin,:,:] = mo_midstep

            self.mo_coeffs[i_spin,:,:] = self.propagate_without_trivial_phase(
                self.old_mo_coeffs[i_spin,:,:], mo_midstep[:,:], mo_tderiv,
                self.init_mo_energies[i_spin], self.t_molecular_orbitals, self.dt, is_initial_step
            )
            #self.mo_coeffs[i_spin,:,:] = self.propagate_with_trivial_phase(
            #    self.old_mo_coeffs[i_spin,:,:], mo_midstep[:,:], mo_tderiv, self.dt, is_initial_step
            #)

            self.old_mo_coeffs[i_spin,:,:] = mo_midstep

            # Debug code
            csc = np.dot( mo_midstep, np.dot( self.S.astype('complex128'), mo_midstep.transpose().conj() ) )
            print('CSC', csc)
            # End Debug code

        self.t_molecular_orbitals += self.dt

        utils.printer.write_out('Updating MOs: Done.\n')

        return


    def propagate_with_trivial_phase(self, old_mo, mid_mo, mo_tderiv, dt, is_init_step):
    
        if is_init_step:
            factor = 1.0
        else:
            factor = 2.0

        new_mo = old_mo + factor * dt * mo_tderiv

        return new_mo

    
    def propagate_without_trivial_phase(self, old_mo, mid_mo, mo_tderiv, init_mo_energies, t, dt, is_init_step):
        
        if is_init_step:
            old_mo_nophase = old_mo * self.get_trivial_phase_factor(init_mo_energies, t   , invert = True)
        else:
            old_mo_nophase = old_mo * self.get_trivial_phase_factor(init_mo_energies, t-dt, invert = True)

        mid_mo_nophase = mid_mo * self.get_trivial_phase_factor(init_mo_energies, t, invert = True)
        
        mo_tderiv_nophase = np.zeros_like(mo_tderiv)

        inv_trivial_phase_factor = self.get_trivial_phase_factor(init_mo_energies, t, invert = True)
        
        for i_MO in range(self.n_MO):

            #mo_tderiv_nophase = (0.0+1.0j) * init_mo_energies * mid_mo_nophase * \
            #                    self.get_trivial_phase_factor(init_mo_energies, t, invert = True) * mo_tderiv
            mo_tderiv_nophase[i_MO,:] = (0.0+1.0j) * init_mo_energies[i_MO] * mid_mo_nophase[i_MO,:] + \
                                inv_trivial_phase_factor[i_MO,:] * mo_tderiv[i_MO,:]

        if is_init_step:
            factor = 1.0
        else:
            factor = 2.0

        new_mo_nophase = old_mo_nophase + factor * dt * mo_tderiv_nophase

        #print('MO_NOPHASE_TEST', new_mo_nophase[0,0]) ## Debug code

        #new_mo = new_mo_nophase * trivial_phase
        new_mo = new_mo_nophase * self.get_trivial_phase_factor(init_mo_energies, t+dt, invert = False)

        return new_mo

    
    def get_trivial_phase_factor(self, mo_energies, t, invert = False):

        trivial_phase = np.zeros( (self.n_MO, self.n_AO), dtype = 'complex128' )
    
        if invert:
            factor = 1.0
        else:
            factor = -1.0

        for i_MO in range(self.n_MO):

            trivial_phase[i_MO,:] = np.exp( factor * (0.0+1.0j) * mo_energies[i_MO] * t)

        return trivial_phase

    
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


    def update_derivative_coupling(self):

        position_2d = utils.coord_1d_to_2d(self.position)
        velocity_2d = utils.coord_1d_to_2d(self.velocity)

        #old_position_2d = position_2d - self.dt * velocity_2d
        #new_position_2d = position_2d + self.dt * velocity_2d
        old_position_2d = position_2d - self.dt_deriv * velocity_2d
        new_position_2d = position_2d + self.dt_deriv * velocity_2d
        
        if self.qc_program == 'dftb+':
            
            overlap_twogeom_0 = dftbplus_manager.worker.return_overlap_twogeom(position_2d,     position_2d)
            overlap_twogeom_1 = dftbplus_manager.worker.return_overlap_twogeom(position_2d, new_position_2d)
            overlap_twogeom_2 = dftbplus_manager.worker.return_overlap_twogeom(position_2d, old_position_2d)

            overlap_twogeom_3 = dftbplus_manager.worker.return_overlap_twogeom(old_position_2d, new_position_2d)

            temp1 = overlap_twogeom_1 - overlap_twogeom_0
            temp2 = overlap_twogeom_2 - overlap_twogeom_0

            temp = np.triu(overlap_twogeom_1 - overlap_twogeom_2)
            #temp = np.triu(overlap_twogeom_3 - overlap_twogeom_0)
            #temp = np.triu(temp1 + temp2)

            overlap_twogeom = temp - temp.transpose()

            #overlap_twogeom = np.triu(overlap_twogeom_1) + np.triu(overlap_twogeom_2).transpose() - np.diag(overlap_twogeom_1)
            

        else:

            utils.stop_with_error("Unknown quantum chemistry program %s .\n" % self.qc_program)

        #self.deriv_coupling = overlap_twogeom / (2.0 * self.dt)
        self.deriv_coupling = overlap_twogeom / (2.0 * self.dt_deriv)

        #print('DERIV_COUPLING', self.deriv_coupling)

        return


    def get_tdnac(self): ## placeholder
        
        n_estate = self.get_n_estate()

        if self.basis == 'configuration':

            # AO -> MO

            # MO -> determinant

            # determinant -> CSF

            pass

        else:

            utils.stop_with_error("Unknown electronic-state basis %s; TDNAC calculation failed.\n" % self.basis)

        tdnac = np.zeros( (n_estate, n_estate), dtype='float64' ) ## placeholder
        
        return tdnac

    
    def get_e_coeffs(self):
        return deepcopy(self.e_coeffs)


    def get_old_e_coeffs(self):
        return deepcopy(self.old_e_coeffs)


    def get_e_coeffs_tderiv(self):
        return deepcopy(self.e_coeffs_tderiv)


    def get_old_e_coeffs_tderiv(self):
        return deepcopy(self.old_e_coeffs_tderiv)


    def get_force(self):
        """Get nuclear force originating from electronic states."""

        force = np.zeros_like(self.position)

        if not self.is_edyn_initialized:
            dftbplus_manager.worker.init_elec_dynamics()
            self.is_edyn_initialized = True

        if self.S is None:
            self.update_matrices()

        S = np.zeros( (1, self.n_AO, self.n_AO), dtype = 'float64' )
        Sinv = np.zeros( (1, self.n_AO, self.n_AO), dtype = 'float64' )
        S[0,:,:] = self.S[:,:]
        Sinv[0,:,:] = self.Sinv[:,:]

        #print('SELF.RHO', self.rho) ## Debug code

        force = dftbplus_manager.worker.get_ehrenfest_force(self.H, self.rho, S, Sinv)

        n_atom = len(self.atomparams)

        force = force.reshape(3*n_atom)

        return force
    

    def update_matrices(self):

        utils.printer.write_out('Updating hamiltonian and overlap matrices: Started.\n')

        if self.qc_program == 'dftb+':
            
            #position_2d = utils.coord_1d_to_2d(self.position) * ANGST2AU
            position_2d = utils.coord_1d_to_2d(self.position)

            dftbplus_manager.worker.set_geometry(position_2d)

            self.update_gs_density_matrix()

            n_AO = sum( dftbplus_manager.worker.get_atom_nr_basis() )

            H = dftbplus_manager.worker.return_hamiltonian(self.gs_rho)
            S = dftbplus_manager.worker.return_overlap_twogeom(position_2d, position_2d)

            n_spin = int(self.is_open_shell) + 1

            self.H = np.zeros_like(H, dtype = 'complex128')

            for i_spin in range(n_spin):
                self.H[i_spin,:,:] = utils.hermitize(H[i_spin,:,:], is_upper_triangle = True)

            self.S = utils.symmetrize(S, is_upper_triangle = True)

            #print('S', S) ## Debug code
            print('S', self.S[0,15], self.S[15,0]) ## Debug code

            self.Sinv = np.linalg.inv(self.S)
            
            #e_vals, e_vecs = sp.eig(self.H[0,:,:], self.S) ## Debug code
            #print('E_VALS', e_vals) ## Debug code
            
            #print('MO_PHASE', self.mo_coeffs) ## Debug code
            #print('MO VALID', np.dot(np.dot(self.mo_coeffs[0,:,:].conjugate(),self.S),self.mo_coeffs[0,:,:].transpose()) ) ## Debug code

            #self.update_molecular_orbitals()
            
            self.update_gs_density_matrix()

            self.construct_density_matrix()

            self.update_derivative_coupling()

            self.update_molecular_orbitals()

            #self.update_gs_density_matrix()

            #n_AO, self.H, self.S = dftbplus_manager.run_dftbplus_text(self.atomparams, self.position)

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


    def construct_density_matrix(self):
        
        if self.basis == 'configuration' and self.excitation == 'cis' and not self.is_open_shell:

            self.rho = np.zeros_like(self.gs_rho, dtype='complex128')

            # Ground-state contribution

            self.rho += self.gs_rho.astype('complex128')

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

            #print('RHO_CIS', rho_cis) ## Debug code

            # MO -> AO

            #rho_cis_ao = 0.0 ## Debug code

            rho_cis_ao = np.dot( active_mos.transpose().conjugate(), np.dot( rho_cis, active_mos ) )

            self.rho[0,:,:] += rho_cis_ao

            print( 'TOTAL NELEC', np.trace( np.dot(self.rho[0,:,:], self.S ) ) ) ## Debug code
            #print( 'EXCITED NELEC', np.trace( np.dot(rho_cis_ao[:,:], self.S ) ) ) ## Debug code
            #print( 'RHO', self.rho[0,:,:] ) ## Debug code

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

    
    def construct_initial_gs(self):

        if self.qc_program == 'dftb+':
        
            n_atom = len(self.atomparams)

            coords = self.position.reshape(n_atom, 3)
            
            dftbplus_manager.worker.set_geometry(coords)

            self.gs_energy = dftbplus_manager.worker.get_energy()

            self.init_mo_energies, mo_coeffs_real = dftbplus_manager.worker.get_molecular_orbitals(
                open_shell = self.is_open_shell
            )

            self.mo_coeffs     = mo_coeffs_real.astype('complex128')
            self.old_mo_coeffs = None

            self.gs_filling = dftbplus_manager.worker.get_filling(open_shell = self.is_open_shell)

            self.n_elec = np.sum(self.gs_filling)

            self.n_MO = np.size(self.mo_coeffs, 2)
            self.n_AO = self.n_MO

            self.update_gs_density_matrix()

            self.rho = self.gs_rho.astype('complex128')

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

                #f[i_MO,i_MO] = self.gs_filling[i_spin][i_MO]
                #scaled_mo_coeffs[:,i_MO] = self.gs_filling[i_spin][i_MO] * self.mo_coeffs[i_spin,:,i_MO]
                scaled_mo_coeffs[i_MO,:] = self.gs_filling[i_spin][i_MO] * self.mo_coeffs[i_spin,i_MO,:]

            rho = np.real(
                #np.dot( self.mo_coeffs[i_spin,:,:], np.transpose(scaled_mo_coeffs) )
                np.dot( np.transpose(self.mo_coeffs[i_spin,:,:]).conjugate(), scaled_mo_coeffs )
            )

            ### Debug code
            #rho[:,:] = 0.0
            #for i_AO in range(self.n_AO):
            #    for j_AO in range(self.n_AO):
            #        for i_MO in range(self.n_MO):
            #            rho[i_AO,j_AO] += self.gs_filling[i_spin][i_MO] * self.mo_coeffs[i_spin,i_MO,i_AO].conjugate() * self.mo_coeffs[i_spin,i_MO,j_AO]
            ### End Debug code

            self.gs_rho[i_spin, :, :] = rho[:, :]
        
        return
