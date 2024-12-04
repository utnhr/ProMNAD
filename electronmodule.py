#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
import scipy.linalg as sp
from copy import deepcopy
from time import perf_counter_ns

from constants import H_DIRAC, ANGST2AU, AU2EV, ONEOVER24
import utils
from settingsmodule import load_setting
from integratormodule import Integrator
from unitarypropagatormodule import UnitaryPropagator
#from interface_dftbplus import dftbplus_manager
from interface_pyscf import pyscf_manager
from calcconfig import init_qc_engine
from basistransformermodule import BasisTransformer
import struct

class Electronic_state:
    
    electronic_state_count = 0
    
    def __init__(self, settings, atomparams, e_coeffs, t_e_coeffs, position, t_position, velocity, t_velocity,
    dtau, alpha, i_step, matrices = None):
        
        self.electronic_state_id = Electronic_state.electronic_state_count
        Electronic_state.electronic_state_count += 1

        self.settings            = deepcopy(settings)

        self.active_occ_mos      = load_setting(settings, 'active_occ_mos')
        self.active_vir_mos      = load_setting(settings, 'active_vir_mos')
        self.qc_program          = load_setting(settings, ('engine', 'type') )

        self.basis               = load_setting(settings, 'basis')
        self.excitation          = load_setting(settings, 'excitation')

        self.reconst_interval    = load_setting(settings, 'reconst_interval')

        self.integmethod         = load_setting(settings, 'integrator')
        #self.do_sc_propagation   = load_setting(settings, 'do_sc_propagation')

        self.propagate_in_orthogonal_basis = load_setting(settings, 'propagate_in_orthogonal_basis')

        self.alpha               = load_setting(settings, 'alpha')
        
        self.e_coeffs            = e_coeffs # np.array (n_estate)
        self.old_e_coeffs        = deepcopy(self.e_coeffs)
        self.t_e_coeffs          = t_e_coeffs
        #### order of linear coefficients ####
        #   case of 'configuration' & 'cis':
        #       

        self.estate_energies     = None
        self.old_estate_energies = None
        self.t_estate_energies   = None

        self.old_position        = deepcopy(position)
        self.position            = deepcopy(position)
        self.t_position          = t_position

        self.old_velocity        = deepcopy(velocity)
        self.velocity            = deepcopy(velocity)
        self.t_velocity          = t_velocity

        self.force               = None
        self.t_force             = None

        self.gs_force            = None
        self.t_gs_force          = None

        self.gs_rho              = None
        self.old_gs_rho          = None

        self.H                   = None
        self.old_H               = None
        self.t_H                 = None

        self.S                   = None
        self.old_S               = None
        self.t_S                 = None

        self.Sinv                = None
        self.old_Sinv            = None
        self.t_Sinv              = None

        self.L                   = None
        self.old_L               = None

        self.deriv_coupling      = None
        self.t_deriv_coupling    = None

        self.tdnac               = None
        self.t_tdnac             = None

        self.dt_deriv            = load_setting(settings, 'dt_deriv')
        
        self.do_interpol         = load_setting(settings, 'do_elec_interpol')
        self.nbin_interpol       = load_setting(settings, 'nbin_interpol_elec')

        self.atomparams          = deepcopy(atomparams)

        self.dtau                = dtau
        self.alpha               = alpha
        self.dt                  = self.alpha * self.dtau

        self.is_open_shell       = False

        self.e_coeffs_tauderiv     = np.zeros_like(e_coeffs)
        self.old_e_coeffs_tauderiv = np.zeros_like(e_coeffs)
        self.t_e_coeffs_tauderiv   = None

        self.i_step              = i_step
        
        ## clocks

        #self.t_mo_coeffs          = None
        #self.t_mo_coeffs_nophase  = None
        #self.t_mo_e_int           = None
        #self.t_mo_levels          = None
        #self.t_csc                = None

        #self.t_H                  = None # TODO
        #self.t_S                  = None # TODO
        ##...

        #

        self.is_edyn_initialized  = False

        #if self.do_sc_propagation:
        #    self.integration_mode = 'sc-propagator'        
        #    self.propagator = UnitaryPropagator(self.integmethod)
        if self.integmethod in UnitaryPropagator.methods:
            self.integration_mode = 'propagator'
            self.propagator = UnitaryPropagator(self.integmethod)
        else:
            self.integration_mode = 'integrator'
            self.integrator = Integrator(self.integmethod, mode = 'chasing')

        #self.gs_energy_integrator = Integrator(self.integmethod)
        #self.e_int_integrator = Integrator(self.integmethod, mode = 'chasing')
        self.e_int_integrator = Integrator('adams_moulton_2', mode = 'chasing') ## Debug code

        self.initial_estate_energies = None

        if self.qc_program == 'dftb+':
            self.dftbplus_instance = init_qc_engine(settings, "%d" % self.electronic_state_id)
        elif self.qc_program == 'pyscf':
            self.pyscf_instance = init_qc_engine(settings, "%d" % self.electronic_state_id)
            self.pyscf_instance_new = init_qc_engine(settings, "%d_newtmp" % self.electronic_state_id) # for numerical AO derivative
            self.pyscf_instance_old = init_qc_engine(settings, "%d_oldtmp" % self.electronic_state_id) # for numerical AO derivative
        else:
            utils.stop_with_error("Unknown QC program %s .\n" % self.qc_program)

        self.reconstruct_gs(is_initial = True)

        if matrices is not None:
            # 'spawned' case
            self.gs_energy           = matrices['gs_energy']
            self.gs_filling          = matrices['gs_filling']
            self.init_mo_energies    = matrices['init_mo_energies']
            self.mo_coeffs           = matrices['mo_coeffs']
            self.mo_coeffs_nophase   = matrices['mo_coeffs_nophase']
            self.old_mo_coeffs       = matrices['old_mo_coeffs']
            self.old_mo_coeffs_nophase = matrices['old_mo_coeffs_nophase']
            self.mo_e_int            = matrices['mo_e_int']
            self.mo_levels           = matrices['mo_levels']
            self.n_elec              = matrices['n_elec']
            self.n_MO                = matrices['n_MO']
            self.n_AO                = matrices['n_AO']
            self.gs_rho              = matrices['gs_rho']
            self.initial_gs_energy   = matrices['initial_gs_energy']
            self.H                   = matrices['H']
            self.S                   = matrices['S']
            self.Sinv                = matrices['Sinv']
            self.tdnac               = matrices['tdnac']
            self.deriv_coupling      = matrices['deriv_coupling']
            self.estate_energies     = matrices['estate_energies']
            self.old_estate_energies = matrices['old_estate_energies']
        
        if self.integration_mode == 'integrator':
            self.initialize_mo_integrator()
        self.initialize_mo_e_int_integrator()

        return


    def set_next_position(self, position, next_t_position):
        self.next_position = deepcopy(position)
        self.next_t_position = next_t_position
        return


    def update_position(self):
        self.old_position  = deepcopy(self.position)
        self.position      = deepcopy(self.next_position)
        self.t_position    = self.next_t_position
        self.next_position = None
        return


    def set_next_velocity(self, velocity, next_t_velocity):
        self.next_velocity = deepcopy(velocity)
        self.next_t_velocity = next_t_velocity
        return


    def update_velocity(self):
        self.old_velocity  = deepcopy(self.velocity)
        self.velocity      = deepcopy(self.next_velocity)
        self.t_velocity    = self.next_t_velocity
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


    def get_e_coeffs_tauderiv(self):
        return deepcopy(self.e_coeffs_tauderiv)


    def get_old_e_coeffs_tauderiv(self):
        return deepcopy(self.old_e_coeffs_tauderiv)


    #def get_molecular_orbitals(self):
    #    """Solve F * C = S * C * e ; get MO energies and coefficients."""    

    #    self.update_molecular_orbitals()

    #    return deepcopy(self.mo_energies), deepcopy(self.mo_coeffs)


    def construct_initial_molecular_orbitals(self):
        """Solve F * C = S * C * e ; get MO energies and coefficients."""    

        utils.check_time_equal(self.t_H, self.t_S)

        self.mo_energies, self.mo_coeffs = sp.eigh(self.H, self.S, type=1)
        #self.mo_energies, self.mo_coeffs = np.linalg.eigh(H, S)
        self.t_mo_levels = self.t_H
        self.t_mo_coeffs = self.t_H
        self.tau_mo_coeffs = self.t_mo_coeffs / self.alpha

        self.mo_coeffs_nophase = deepcopy(self.mo_coeffs) # initial MOs are real
        self.t_mo_coeffs_nophase = self.t_mo_coeffs

        #self.t_mos = self.get_istep()

        return


    #def make_mo_tderiv_old(self, t, mo_coeffs, deriv_coupling, Sinv, without_trivial_phase = True, init_mo_energies = None):

    #    if self.is_open_shell:
    #        n_spin = 2
    #    else:
    #        n_spin = 1

    #    trivial_phase_factor     = self.get_trivial_phase_factor(init_mo_energies, t, invert = False)
    #    inv_trivial_phase_factor = self.get_trivial_phase_factor(init_mo_energies, t, invert = True)

    #    def get_mo_dependent_gs_density_matrix(mo_coeffs):
    #        
    #        # mo_coeffs can either phase-containing or phaseless MOs
    #        
    #        gs_rho = np.zeros( (n_spin, self.n_AO, self.n_AO), dtype = 'float64' )

    #        for i_spin in range(n_spin):

    #            scaled_mo_coeffs = np.zeros_like(mo_coeffs[i_spin,:,:])

    #            for i_MO in range(self.n_MO):

    #                scaled_mo_coeffs[i_MO,:] = self.gs_filling[i_spin][i_MO] * mo_coeffs[i_spin,i_MO,:]

    #            rho = np.dot( np.transpose(mo_coeffs[i_spin,:,:]).conjugate(), scaled_mo_coeffs )

    #            gs_rho[i_spin, :, :] = np.real(rho[:, :])

    #        #print('RHO 0 4', gs_rho[0,0,4]) ## Debug code
    #        
    #        return gs_rho
    #    
    #    def get_mo_dependent_hamiltonian(mo_coeffs):
    #        
    #        # mo_coeffs can either phase-containing or phaseless MOs

    #        gs_rho = get_mo_dependent_gs_density_matrix(mo_coeffs)
    #        
    #        self.dftbplus_instance.go_to_workdir()
    #        Htmp = self.dftbplus_instance.worker.return_hamiltonian(gs_rho)
    #        self.dftbplus_instance.return_from_workdir()

    #        H = np.zeros_like(Htmp, dtype = 'complex128')

    #        for i_spin in range(n_spin):
    #            H[i_spin,:,:] = utils.hermitize(Htmp[i_spin,:,:], is_upper_triangle = True)
    #        
    #        return H

    #    if without_trivial_phase:
    #        mo = mo_coeffs * trivial_phase_factor
    #    else:
    #        mo = mo_coeffs

    #    H = get_mo_dependent_hamiltonian(mo)

    #    Heff = np.zeros_like(H)

    #    mo_tderiv = np.zeros_like(mo)

    #    for i_spin in range(n_spin):

    #        Heff[i_spin,:,:] = H[i_spin,:,:] - (0.0+1.0j) * deriv_coupling[:,:]
    #    
    #        mo_tderiv[i_spin,:,:] = -(0.0+1.0j) * np.dot(
    #            np.dot( Sinv.astype('complex128'), Heff[i_spin,:,:] ), mo[i_spin,:,:].transpose()
    #        ).transpose()

    #    if without_trivial_phase:

    #        for i_spin in range(n_spin):
    #            for i_MO in range(self.n_MO):

    #                mo_tderiv[i_spin,i_MO,:] = (0.0+1.0j) * init_mo_energies[i_spin,i_MO] * \
    #                    mo_coeffs[i_spin,i_MO,:] + inv_trivial_phase_factor[i_spin,i_MO,:] * mo_tderiv[i_spin,i_MO,:]

    #    return mo_tderiv


    def get_mo_dependent_gs_density_matrix(self, mo_coeffs):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        # mo_coeffs can either phase-containing or phaseless MOs
        
        gs_rho = np.zeros( (n_spin, self.n_AO, self.n_AO), dtype = 'float64' )

        for i_spin in range(n_spin):

            scaled_mo_coeffs = np.zeros_like(mo_coeffs[i_spin,:,:])

            for i_MO in range(self.n_MO):

                scaled_mo_coeffs[i_MO,:] = self.gs_filling[i_spin][i_MO] * mo_coeffs[i_spin,i_MO,:]

            rho = np.dot( np.transpose(mo_coeffs[i_spin,:,:]).conjugate(), scaled_mo_coeffs )

            gs_rho[i_spin, :, :] = np.real(rho[:, :])

        #print('RHO 0 4', gs_rho[0,0,4]) ## Debug code
        
        return gs_rho


    def get_mo_dependent_hamiltonian(self, mo_coeffs, rho=None):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        # mo_coeffs can either phase-containing or phaseless MOs
        # optionally, rho can be given instead of mo_coeffs

        if rho is None:

            gs_rho = self.get_mo_dependent_gs_density_matrix(mo_coeffs)

        else:

            gs_rho = deepcopy(rho)

        if self.qc_program == 'dftb+':
        
            self.dftbplus_instance.go_to_workdir()
            Htmp = self.dftbplus_instance.worker.return_hamiltonian(gs_rho)
            self.dftbplus_instance.return_from_workdir()

        elif self.qc_program == 'pyscf':
            
            if not n_spin == 1:
                utils.stop_with_error("Currently not compatible with open-shell systems.")
            
            Htmp = self.pyscf_instance.return_hamiltonian(gs_rho[0,:,:], n_spin)

        else:

            utils.stop_with_error("Not compatible with quantum chemistry program %s ." % self.qc_program)

        H = np.zeros_like(Htmp, dtype = 'complex128')

        for i_spin in range(n_spin):
            H[i_spin,:,:] = utils.hermitize(Htmp[i_spin,:,:], is_upper_triangle = True)
        
        return H


    def make_mo_nophase_tauderiv(self, tau, mo_coeffs_nophase, deriv_coupling, Sinv, init_mo_energies = None):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        #def get_mo_dependent_gs_density_matrix(mo_coeffs):
        #    
        #    # mo_coeffs can either phase-containing or phaseless MOs
        #    
        #    gs_rho = np.zeros( (n_spin, self.n_AO, self.n_AO), dtype = 'float64' )

        #    for i_spin in range(n_spin):

        #        scaled_mo_coeffs = np.zeros_like(mo_coeffs[i_spin,:,:])

        #        for i_MO in range(self.n_MO):

        #            scaled_mo_coeffs[i_MO,:] = self.gs_filling[i_spin][i_MO] * mo_coeffs[i_spin,i_MO,:]

        #        rho = np.dot( np.transpose(mo_coeffs[i_spin,:,:]).conjugate(), scaled_mo_coeffs )

        #        gs_rho[i_spin, :, :] = np.real(rho[:, :])

        #    #print('RHO 0 4', gs_rho[0,0,4]) ## Debug code
        #    
        #    return gs_rho
        
        #def get_mo_dependent_hamiltonian(mo_coeffs):
        #    
        #    # mo_coeffs can either phase-containing or phaseless MOs

        #    gs_rho = get_mo_dependent_gs_density_matrix(mo_coeffs)
        #    
        #    self.dftbplus_instance.go_to_workdir()
        #    Htmp = self.dftbplus_instance.worker.return_hamiltonian(gs_rho)
        #    self.dftbplus_instance.return_from_workdir()

        #    H = np.zeros_like(Htmp, dtype = 'complex128')

        #    for i_spin in range(n_spin):
        #        H[i_spin,:,:] = utils.hermitize(Htmp[i_spin,:,:], is_upper_triangle = True)
        #    
        #    return H

        ### Debug code
        #print("SVD TEST")
        #P = BasisTransformer.in_new_basis(self.S, mo_coeffs_nophase[0])
        #print(P)
        #sys.exit()
        ### End Debug code
        
        if self.do_interpol:
            H_full = deepcopy(self.H)
        else:
            H_full = self.get_mo_dependent_hamiltonian(mo_coeffs_nophase)

        mo_nophase_tauderiv = np.zeros_like(mo_coeffs_nophase)

        self.Heff = np.zeros_like(H_full)

        for i_spin in range(n_spin):
            
            #print('DERIVCOUPL', deriv_coupling) ## Debug code
            
            C = mo_coeffs_nophase[i_spin]
            H = H_full[i_spin,:,:]

            H_nophase_ao = self.make_H_nophase(H, self.S, C)

            Heff_nophase = H_nophase_ao - (0.0+1.0j) * deriv_coupling[:,:]
            self.Heff[i_spin,:,:] = H - (0.0+1.0j) * deriv_coupling[:,:] # for reuse in the later steps

            mo_nophase_tauderiv[i_spin,:,:] = -(0.0+1.0j) * np.dot(
                np.dot( Sinv.astype('complex128'), Heff_nophase ), C.transpose()
            ).transpose()

        return mo_nophase_tauderiv


    def make_mo_nophase_tauderiv_ort(self, tau, mo_coeffs_nophase_ort):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        mo_coeffs_nophase = np.zeros_like(mo_coeffs_nophase_ort)

        for i_spin in range(n_spin):
            # project onto basis at current nuclear coordinate
            mo_coeffs_nophase[i_spin,:,:] = BasisTransformer.moinao(mo_coeffs_nophase_ort[i_spin,:,:], self.S, self.L)

        #test = mo_coeffs_nophase - self.mo_coeffs_nophase ## Debug code
        #test = mo_coeffs_nophase_ort ## Debug code
        #print('TEST2', test) ## Debug code

        if self.do_interpol:
            H_full = deepcopy(self.H)
        else:
            H_full = self.get_mo_dependent_hamiltonian(mo_coeffs_nophase)

        mo_nophase_tauderiv_ort = np.zeros_like(mo_coeffs_nophase_ort)

        self.Heff = np.zeros_like(H_full)

        for i_spin in range(n_spin):
            
            C = mo_coeffs_nophase_ort[i_spin,:,:]
            C_ao = mo_coeffs_nophase[i_spin,:,:]
            #C_ao = self.mo_coeffs_nophase[i_spin,:,:]; print('FOR DEBUG') ## Debug code
            
            H_nophase_ao = self.make_H_nophase(H_full[i_spin,:,:], self.S, C_ao)

            #Heff_nophase = deepcopy(H_nophase_ao)
            self.Heff = deepcopy(H_full) # for reuse in the later steps

            H_nophase = BasisTransformer.ao2mo(H_nophase_ao, self.L)

            mo_nophase_tauderiv_ort[i_spin,:,:] = -(0.0+1.0j) * np.dot( H_nophase, C.transpose() ).transpose()

        return mo_nophase_tauderiv_ort


    def make_H_nophase(self, H, S, C):
        
            print('PHASE EXTRACTION DISABLED') ## Debug code
            return deepcopy(H) ## Debug code

            ## AO -> MO

            #H_nophase_mo = np.dot( C.conjugate(), np.dot( H, C.transpose() ) )

            ## subtract MO energy
            #
            ##for i_MO in range(self.n_MO):
            ##    H_nophase_mo[i_MO,i_MO] = 0.0+0.0j

            ##print('H_NOPHASE_MO', H_nophase_mo) ## Debug code

            ## back to AO

            #SC = np.dot( self.S, C.transpose() )

            #H_nophase_ao = np.dot( SC, np.dot( H_nophase_mo, SC.transpose().conjugate() ) )

            #return H_nophase_ao

    
    # call after make_mo_nophase_tderiv
    # (to make sure that self.Heff is up to date)
    def make_mo_e_int_tauderiv(self, tau, mo_e_int, Heff):

        mo_e_int_tauderiv = np.zeros_like(mo_e_int)

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        for i_spin in range(n_spin):

            mo_H = np.dot(
                self.mo_coeffs_nophase[i_spin,:,:].conjugate(), np.dot( Heff[i_spin,:,:], self.mo_coeffs_nophase[i_spin,:,:].transpose() )
            )

            #mo_e_int_tderiv[i_spin,:] = np.diag(Heff[i_spin,:,:])
            mo_e_int_tauderiv[i_spin,:] = np.diag(mo_H)

        return mo_e_int_tauderiv


    def update_csc(self):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        utils.check_time_equal(self.t_mo_coeffs, self.t_S)

        for i_spin in range(n_spin):

            mo_midstep = deepcopy(self.mo_coeffs[i_spin,:,:])

            csc = np.dot( mo_midstep, np.dot( self.S.astype('complex128'), mo_midstep.transpose().conj() ) )

            self.csc = np.real(np.diag(csc))

        self.t_csc = self.t_mo_coeffs
    
    
    def update_mo_tdnac(self):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        utils.check_time_equal(self.t_mo_coeffs, self.t_position)
        utils.check_time_equal(self.t_mo_coeffs, self.t_velocity)

        ## (1/2) * <\psi(t-dt)|\psi(t+dt)>

        #position_2d = utils.coord_1d_to_2d(self.position)
        #velocity_2d = utils.coord_1d_to_2d(self.velocity)

        ##old_position_2d = position_2d - self.dt * velocity_2d
        #old_position_2d = position_2d - 2.0 * self.dt * velocity_2d
        ##new_position_2d = position_2d + self.dt * velocity_2d
        #
        #if self.qc_program == 'dftb+':
        #    
        #    self.dftbplus_instance.go_to_workdir()
        #    overlap_twogeom_1 = self.dftbplus_instance.worker.return_overlap_twogeom(old_position_2d, position_2d)
        #    overlap_twogeom_2 = self.dftbplus_instance.worker.return_overlap_twogeom(position_2d, old_position_2d)
        #    #overlap_twogeom_1 = self.dftbplus_instance.worker.return_overlap_twogeom(old_position_2d, new_position_2d)
        #    #overlap_twogeom_2 = self.dftbplus_instance.worker.return_overlap_twogeom(new_position_2d, old_position_2d)
        #    self.dftbplus_instance.return_from_workdir()

        #    #S = np.triu(overlap_twogeom_1) + np.triu(overlap_twogeom_2).transpose() - np.diag(np.diag(overlap_twogeom_1))

        #else:

        #    utils.stop_with_error("MO TDNAC calculation is not compatible with QC program %s .\n" % self.qc_program)

        #temp = np.triu(overlap_twogeom_1) + np.triu(overlap_twogeom_2).transpose() - np.diag( np.diag(overlap_twogeom_2) )

        self.mo_tdnac = np.zeros_like(self.mo_coeffs)
        
        if self.do_interpol:
            
            H = deepcopy(self.H)

        else:

            H = self.get_mo_dependent_hamiltonian(self.mo_coeffs)

        Heff = np.zeros_like(H)

        for i_spin in range(n_spin):

            Heff[i_spin,:,:] = H[i_spin,:,:] - (0.0+1.0j) * self.deriv_coupling[:,:]
            
            # According to TDKS
            # <p|d/dt|q> = (-i / \hbar) * <p|Heff|q>
            # overlap matrix ??

            #self.mo_tdnac[i_spin,:,:] = 0.5 * np.dot(
            #    np.dot(np.conj(self.old_mo_coeffs[i_spin,:,:]), temp.astype('complex128')),
            #    new_mo_coeffs[i_spin,:,:].transpose()
            #) / self.dt
            #self.mo_tdnac[i_spin,:,:] = np.dot(
            #    np.dot(np.conj(self.mo_coeffs[i_spin,:,:]), self.deriv_coupling.astype('complex128')),
            #    self.mo_coeffs[i_spin,:,:].transpose()
            #)
            self.mo_tdnac[i_spin,:,:] = -(0.0+1.0j) * np.dot(
                np.dot(np.conj(self.mo_coeffs[i_spin,:,:]), Heff[i_spin,:,:]),
                self.mo_coeffs[i_spin,:,:].transpose()
            )

        self.t_mo_tdnac = self.t_mo_coeffs

        #print('MO OVERLAP TWOGEOM', self.mo_tdnac * 2.0 * self.dt) ## Debug code
        #print( np.dot( np.dot( np.conj(self.mo_coeffs[0,:,:]), H[0,:,:]), self.mo_coeffs[0,:,:].transpose() ) ) ## Debug code

        return


    def propagate_molecular_orbitals(self):
        """Update MO energies and coefficients according to TD-KS equation."""
        #utils.Printer.write_out('Updating MOs: Started.\n')

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        self.old_mo_coeffs = deepcopy(self.mo_coeffs)

        #utils.check_time_equal(self.t_mo_coeffs, self.t_S)

        dtau = self.dtau / float(self.nbin_interpol)
        dt = self.dt / float(self.nbin_interpol)

        for ibin_interpol in range(1, self.nbin_interpol+1):

            if ibin_interpol == 1:
                is_first_call = True
            else:
                is_first_call = False

            self.interpolate_matrices_and_nuclei(ibin_interpol, self.nbin_interpol, is_first_call)
            
            ts = perf_counter_ns()
            new_mo_coeffs = self.propagate_without_trivial_phase(self.tau_mo_coeffs, dtau, dt)
            te = perf_counter_ns()
            print("Time for MO integration: %12.3f sec.\n" % ((te-ts)/1.0e+9)) ## Debug code

            self.mo_coeffs = deepcopy(new_mo_coeffs)
            #self.t_mo_coeffs += dt

        self.t_mo_coeffs += self.dt
        self.tau_mo_coeffs += self.dtau
        
        #utils.Printer.write_out('Updating MOs: Done.\n')

        return


    #def propagate_with_trivial_phase(self, old_mo, mid_mo, mo_tderiv, dt, is_init_step):
    #
    #    if is_init_step:
    #        factor = 1.0
    #    else:
    #        factor = 2.0

    #    new_mo = old_mo + factor * dt * mo_tderiv

    #    return new_mo


    def initialize_mo_integrator(self):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        if self.propagate_in_orthogonal_basis:

            mo_coeffs_nophase_ort = np.zeros_like(self.mo_coeffs_nophase)

            for i_spin in range(n_spin):
                mo_coeffs_nophase_ort[i_spin,:,:] = BasisTransformer.moinmo(
                    self.mo_coeffs_nophase[i_spin,:,:], self.S, self.L,
                )

            self.integrator.initialize_history(
                self.t_mo_coeffs_nophase, mo_coeffs_nophase_ort,
                self.make_mo_nophase_tauderiv_ort,
            )

        else:
        
            self.integrator.initialize_history(
                self.t_mo_coeffs_nophase, self.mo_coeffs_nophase,
                self.make_mo_nophase_tauderiv, self.deriv_coupling, self.Sinv,
            )


    def initialize_mo_e_int_integrator(self):

        self.e_int_integrator.initialize_history(
            self.t_mo_e_int, self.mo_e_int,
            self.make_mo_e_int_tauderiv, self.H,
        )


    
    def propagate_without_trivial_phase(self, tau, dtau, dt):
        
        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        self.old_mo_coeffs_nophase = deepcopy(self.mo_coeffs_nophase)
        self.old_mo_e_int          = deepcopy(self.mo_e_int)

        #utils.check_time_equal(self.t_mo_coeffs_nophase, self.t_deriv_coupling)
        #utils.check_time_equal(self.t_mo_coeffs_nophase, self.t_Sinv)

        if self.integration_mode == 'integrator':

            if self.propagate_in_orthogonal_basis:

                mo_coeffs_nophase_ort = np.zeros_like(self.mo_coeffs_nophase)

                # At this point, self.mo_coeffs_nophase is based on the previous geometry
                for i_spin in range(n_spin):
                    mo_coeffs_nophase_ort[i_spin,:,:] = BasisTransformer.moinmo(
                        self.mo_coeffs_nophase[i_spin,:,:], self.old_S, self.old_L,
                    )

                    ## Debug code
                    ##test = BasisTransformer.moinao(mo_coeffs_nophase_ort[i_spin,:,:],self.S,self.L) - self.mo_coeffs_nophase[i_spin,:,:]
                    #test = mo_coeffs_nophase_ort
                    #print('TEST1', test)
                    ## End Debug code
                
                # propagate in orthonormalized basis
                mo_coeffs_nophase_ort = self.integrator.engine(
                    dtau, self.tau_mo_coeffs_nophase, mo_coeffs_nophase_ort,
                    self.make_mo_nophase_tauderiv_ort,
                )
                
                # project onto current geometry
                for i_spin in range(n_spin):
                    self.mo_coeffs_nophase[i_spin,:,:] = BasisTransformer.moinao(
                        mo_coeffs_nophase_ort[i_spin,:,:], self.S, self.L
                    )

            else:
        
                self.mo_coeffs_nophase = self.integrator.engine(
                    #self.dt, self.t_mo_coeffs_nophase, self.mo_coeffs_nophase,
                    dtau, self.tau_mo_coeffs_nophase, self.mo_coeffs_nophase,
                    self.make_mo_nophase_tauderiv, self.deriv_coupling, self.Sinv,
                )

        elif self.integration_mode == 'propagator':

            self.Heff = np.zeros_like(self.H)
            self.Heff_nophase = np.zeros_like(self.H)

            if n_spin > 1:

                utils.stop_with_error('Propagator not compatible with open-shell systems')

            i_spin = 0

            if self.propagate_in_orthogonal_basis:

                mo_coeffs_nophase_ort = np.zeros_like(self.mo_coeffs_nophase)

                # At this point, self.mo_coeffs_nophase is based on the previous geometry
                for i_spin in range(n_spin):
                    mo_coeffs_nophase_ort[i_spin,:,:] = BasisTransformer.moinmo(
                        self.mo_coeffs_nophase[i_spin,:,:], self.old_S, self.old_L,
                    )

                # propagate in orthonormalized basis
                #self.mo_coeffs_nophase[i_spin,:,:] = self.propagator.engine(
                mo_coeffs_nophase_ort[i_spin,:,:] = self.propagator.engine(
                    dtau, mo_coeffs_nophase_ort[i_spin,:,:], self.get_H_orthogonal_nophase_nospin,
                )

                # project onto current geometry
                for i_spin in range(n_spin):
                    self.mo_coeffs_nophase[i_spin,:,:] = BasisTransformer.moinao(
                        mo_coeffs_nophase_ort[i_spin,:,:], self.S, self.L
                    )

            else:

                self.mo_coeffs_nophase[i_spin,:,:] = self.propagator.engine(
                    dtau, self.mo_coeffs_nophase[i_spin,:,:], self.get_SinvHeff_nophase_nospin, True,
                )

                #H = self.H[i_spin,:,:]
                #H_nophase = self.make_H_nophase(H, self.S, self.mo_coeffs[i_spin,:,:])

                #self.Heff[i_spin,:,:] = H - (0.0+1.0j) * self.deriv_coupling[:,:]
                #self.Heff_nophase[i_spin,:,:] = H_nophase - (0.0+1.0j) * self.deriv_coupling[:,:]

                #ts = perf_counter_ns()
                #self.mo_coeffs_nophase[i_spin,:,:] = self.propagator.engine(
                #    #self.mo_coeffs[i_spin,:,:], self.Sinv, self.Heff[i_spin,:,:], dtau
                #    self.mo_coeffs_nophase[i_spin,:,:], self.Sinv, self.Heff_nophase[i_spin,:,:], dtau
                #)
                #te = perf_counter_ns()
                #print("Time for propagator: %12.3f sec.\n" % ((te-ts)/1.0e+9)) ## Debug code

        #elif self.integration_mode == 'sc-propagator':

        #    self.Heff = np.zeros_like(self.H)
        #    self.Heff_nophase = np.zeros_like(self.H)

        #    threshold = 1.0e-11
        #    mix_param = 0.3

        #    error = threshold + 1 # larger than threshold

        #    H_spin = deepcopy(self.H)
        #    mo_coeffs_nophase = deepcopy(self.mo_coeffs_nophase)
        #    new_mo_coeffs_nophase = np.zeros_like(mo_coeffs_nophase)
        #    old_rho = None

        #    i_cycle = 0
        #    
        #    while error > threshold:

        #        print('SC CYCLE: ', i_cycle) ## Debug code

        #        for i_spin in range(n_spin):

        #            H_nophase = self.make_H_nophase(H_spin[i_spin,:,:], self.S, mo_coeffs_nophase[i_spin,:,:])

        #            Heff_nophase = H_nophase - (0.0+1.0j) * self.deriv_coupling[:,:]

        #            new_mo_coeffs_nophase[i_spin,:,:] = self.propagator.engine(
        #                mo_coeffs_nophase[i_spin,:,:], self.Sinv, Heff_nophase, dtau
        #            )

        #        rho = self.get_mo_dependent_gs_density_matrix(new_mo_coeffs_nophase)

        #        if i_cycle == 0:
        #            
        #            old_rho = deepcopy(rho)

        #        else:

        #            error = (np.abs(rho - old_rho)).max()

        #            rho = mix_param * rho + (1 - mix_param) * old_rho

        #            old_rho = deepcopy(rho)

        #        H_spin = self.get_mo_dependent_hamiltonian(mo_coeffs=None, rho=rho)

        #        print(error)

        #        i_cycle += 1

        #    self.mo_coeffs_nophase = deepcopy(new_mo_coeffs_nophase)

        #    for i_spin in range(n_spin):

        #        H_nophase = self.make_H_nophase(self.H[i_spin,:,:], self.S, mo_coeffs_nophase[i_spin,:,:])
        #        
        #        self.Heff_nophase[i_spin,:,:] = H_nophase - (0.0+1.0j) * self.deriv_coupling[:,:]
        #        self.Heff[i_spin,:,:] = self.H[i_spin,:,:] - (0.0+1.0j) * self.deriv_coupling[:,:]

        else:
            
            utils.stop_with_error("Unknown integration mode %s ." % self.integration_mode)

        self.t_mo_coeffs_nophase += dt
        self.tau_mo_coeffs_nophase += dtau

        self.mo_e_int = self.e_int_integrator.engine(
            self.dtau, self.tau_mo_e_int, self.mo_e_int,
            self.make_mo_e_int_tauderiv, self.Heff,
        )
        self.t_mo_e_int += dt
        self.tau_mo_e_int += dtau

        #if self.integration_mode == 'integrator':
        
        print('NO ADDITIONAL PHASE FACTOR FOR DEBUG') ## Debug code
        new_mo = deepcopy(self.mo_coeffs_nophase) ## Debug code
        #new_mo = np.zeros_like(self.mo_coeffs_nophase)

        #for i_spin in range(n_spin):

        #    for i_MO in range(self.n_MO):
        #            
        #        new_mo[i_spin,i_MO,:] = \
        #            self.mo_coeffs_nophase[i_spin,i_MO,:] * \
        #            np.exp( (-1.0j/H_DIRAC) * self.mo_e_int[i_spin,i_MO] )

        return new_mo


    def get_SinvHeff_nophase_nospin(self, mo_coeffs, update_selfH):

        H = self.get_mo_dependent_hamiltonian( np.array([mo_coeffs]) )[0] # i_spin == 0

        #print('PHASE SUBTRACTION DISABLED FOR DEBUG') ## Debug code
        H_nophase = self.make_H_nophase(H, self.S, mo_coeffs)
        #H_nophase = deepcopy(H) ## Debug code

        Heff_nophase = H_nophase - (0.0+1.0j) * self.deriv_coupling[:,:]

        i_spin = 0

        if update_selfH:

            self.H[i_spin,:,:] = deepcopy(H)
            self.Heff[i_spin,:,:] = H - (0.0+1.0j) * self.deriv_coupling[:,:]
            self.Heff_nophase[i_spin,:,:] = deepcopy(Heff_nophase)

        return np.dot(self.Sinv, Heff_nophase)


    def get_H_orthogonal_nophase_nospin(self, mo_coeffs_ort_nospin):
        
        mo_coeffs = BasisTransformer.moinao(mo_coeffs_ort_nospin, self.S, self.L)

        H = self.get_mo_dependent_hamiltonian( np.array([mo_coeffs]) )[0] # i_spin == 0

        i_spin = 0

        H_nophase = self.make_H_nophase(H, self.S, mo_coeffs)

        return BasisTransformer.ao2mo(H_nophase, self.L)

    
    def get_trivial_phase_factor(self, mo_energies, tau, invert = False):

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

                trivial_phase[i_spin,i_MO,:] = np.exp( factor * (0.0+1.0j) * mo_energies[i_spin,i_MO] * tau)

        return trivial_phase


    #def make_gs_energy_tderiv(self, t, y):

    #    #if self.gs_force is None or self.old_velocity is None:
    #    #    return 0.0
    #    
    #    return -np.dot(self.gs_force, self.old_velocity)

    
    def update_estate_energies(self):
        """Get energy of each 'electronic state', which is i->a excitation configuration. Approximate state energy as MO energy difference."""

        #utils.Printer.write_out('Updating electronic state energies: Started.\n')

        if self.is_open_shell:

            utils.stop_with_error('Currently not compatible with open-shell systems.\n')

        else:

            utils.check_time_equal(self.t_mo_levels, self.t_gs_energy)

            #self.gs_energy = self.gs_energy_integrator.engine(self.dt, 0.0, self.gs_energy, self.make_gs_energy_tderiv)
            #self.t_gs_energy += self.dt

            mo_energies = self.mo_levels[0,:]
            #mo_coeffs   = self.mo_coeffs[0,:,:]

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
            self.t_estate_energies = self.t_mo_levels

            #utils.Printer.write_out('Updating electronic state energies: Done.\n')

            return


    def deliver(self, i_estate):
        
        self.e_coeffs[i_estate] = 0.0+0.0j
        
        new_norm = np.linalg.norm(self.e_coeffs)

        self.e_coeffs /= new_norm

        self.e_coeffs_tauderiv[i_estate] = 0.0+0.0j

        self.e_coeffs_tauderiv /= new_norm

        return


    def get_estate_energies(self):

        #if self.estate_energies is None:

        #    self.update_gs_hamiltonian_and_molevels()

        #    self.update_estate_energies()

        return deepcopy(self.estate_energies)


    def update_ehrenfest_energy(self):

        # Ehrenfest energy: E = <\Psi|H|\Psi>
        # Here H assumed to be diagonal, so E is just a weighted average
        
        val = 0.0

        n_estate = len(self.e_coeffs)

        utils.check_time_equal(self.t_estate_energies, self.t_e_coeffs)

        for i_estate in range(n_estate):
            
            val += self.estate_energies[i_estate] * abs(self.e_coeffs[i_estate])**2

        self.ehrenfest_energy = val

        self.t_ehrenfest_energy = self.t_estate_energies

        return


    def get_ehrenfest_energy(self):
        
        return self.ehrenfest_energy


    def update_derivative_coupling(self):

        if self.deriv_coupling is not None:
            self.old_deriv_coupling = deepcopy(self.deriv_coupling)

        utils.check_time_equal(self.t_position, self.t_velocity)

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

            #self.deriv_coupling = overlap_twogeom / self.dt
            #self.deriv_coupling = overlap_twogeom / (2.0 * self.dt)
            self.deriv_coupling = overlap_twogeom / (2.0 * self.dt_deriv)
            self.t_deriv_coupling = self.t_position

        elif self.qc_program == 'pyscf':
            
            #self.deriv_coupling = self.pyscf_instance.get_derivative_coupling(velocity_2d)

            instance_new = self.pyscf_instance_new
            instance_old = self.pyscf_instance_old
            #instance_old = pyscf_manager(self.settings, self.pyscf_instance.mol.atom)

            instance_new.update_geometry(new_position_2d)
            instance_old.update_geometry(old_position_2d)

            mol_new = instance_new.mol
            mol_old = instance_old.mol
            mol     = self.pyscf_instance.mol

            overlap_twogeom_0 = self.pyscf_instance.gto.intor_cross('int1e_ovlp', mol    , mol    )
            overlap_twogeom_1 = self.pyscf_instance.gto.intor_cross('int1e_ovlp', mol    , mol_new)
            overlap_twogeom_2 = self.pyscf_instance.gto.intor_cross('int1e_ovlp', mol    , mol_old)
            overlap_twogeom_3 = self.pyscf_instance.gto.intor_cross('int1e_ovlp', mol_new, mol    )
            overlap_twogeom_4 = self.pyscf_instance.gto.intor_cross('int1e_ovlp', mol_old, mol    )

            temp1 = np.triu(overlap_twogeom_1) + np.triu(overlap_twogeom_3).transpose()
            temp2 = np.triu(overlap_twogeom_2) + np.triu(overlap_twogeom_4).transpose()

            temp = temp1 - temp2

            overlap_twogeom = temp

            self.deriv_coupling = overlap_twogeom / (2.0 * self.dt_deriv)
            self.t_deriv_coupling = self.t_position

        else:

            utils.stop_with_error("Unknown quantum chemistry program %s .\n" % self.qc_program)

        return


    def update_tdnac(self): # TDNAC at t-(1/2)dt
        
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

            S_occ = np.zeros( (n_spin, n_occ, n_occ), dtype = 'complex128' ) # for active occ. MOs
            S_vir = np.zeros( (n_spin, n_vir, n_vir), dtype = 'complex128' ) # for active vir. MOs

            for i_spin in range(n_spin):

                S_occ[i_spin,:,:] = self.mo_tdnac[i_spin,self.active_occ_mos,:][:,self.active_occ_mos]
                S_vir[i_spin,:,:] = self.mo_tdnac[i_spin,self.active_vir_mos,:][:,self.active_vir_mos]

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
                    tdnac[j_estate, k_estate] = -np.conj(val)
                
                    ### Debug code
                    #if k_estate == 1 and j_estate == 9:
                    #    print('TDNAC between 1/9:', val)
                    ### End Debug code

            #for i_occ in range(n_occ):
            #        tdnac[ 1 + i_occ * n_vir : , 1 + i_occ * n_vir : ] = nac_vir[:, :]

            self.tdnac = tdnac
            self.t_tdnac = self.t_mo_tdnac

        else:

            utils.stop_with_error("Unknown electronic-state basis %s; TDNAC calculation failed.\n" % self.basis)
        
        return


    def get_tdnac(self):

        if self.tdnac is None:

            n_estate = self.get_n_estate()

            self.tdnac = np.zeros( (n_estate, n_estate), dtype='complex128' )

        return self.tdnac

    
    def get_force(self, gs_force = False, do_not_update = False, get_time = False):
        """Get nuclear force originating from electronic states."""

        force = np.zeros_like(self.position)

        if self.qc_program == 'dftb+':

            if not self.is_edyn_initialized:
                self.dftbplus_instance.go_to_workdir()
                self.dftbplus_instance.worker.init_elec_dynamics()
                self.dftbplus_instance.return_from_workdir()
                self.is_edyn_initialized = True

            #if self.S is None:
            #    self.update_matrices(is_before_initial = True)
            
            utils.check_time_equal(self.t_S, self.t_Sinv)
            S = np.zeros( (1, self.n_AO, self.n_AO), dtype = 'float64' )
            Sinv = np.zeros( (1, self.n_AO, self.n_AO), dtype = 'float64' )
            S[0,:,:] = self.S[:,:]
            Sinv[0,:,:] = self.Sinv[:,:]
            
            if gs_force:
                utils.check_time_equal(self.t_gs_rho, self.t_S)
                rho_real = self.gs_rho
            else:
                utils.check_time_equal(self.t_rho, self.t_S)
                rho_real = np.zeros_like(self.rho, dtype = 'float64')
                rho_real[:,:,:] = self.rho[:,:,:]
            
            H = np.zeros_like(self.rho)
            self.dftbplus_instance.go_to_workdir()
            tmp = self.dftbplus_instance.worker.return_hamiltonian(rho_real) # Internal atom positions ??
            self.dftbplus_instance.return_from_workdir()

            if self.is_open_shell:
                n_spin = 2
            else:
                n_spin = 1

            for i_spin in range(n_spin):
                H[i_spin,:,:] = utils.hermitize(tmp[i_spin,:,:], is_upper_triangle = True)

            self.dftbplus_instance.go_to_workdir()
            force = self.dftbplus_instance.worker.get_ehrenfest_force(H, self.rho, S, Sinv)
            self.dftbplus_instance.return_from_workdir()

            n_atom = len(self.atomparams)

            force = force.reshape(3*n_atom)

        elif self.qc_program == 'pyscf':
            
            print('WARNING: GROUND-STATE FORCE') ## Debug code

            self.force = self.pyscf_instance.ks.Gradients()

        else:

            utils.stop_with_error("Unknown QC program %s ." % self.qc_program)

        if not do_not_update:
            
            if gs_force:
                self.gs_force = force
                self.t_gs_force = self.t_S

            else:
                self.force = force
                self.t_force = self.t_S
        
        if get_time and gs_force:
            return force, self.t_gs_force
        elif get_time and not gs_force:
            return force, self.t_force
        else:
            return force
    

    def update_position_dependent_quantities(self):

        # Update stuffs that depend on position and velocity, but not on MOs

        self.old_position = deepcopy(self.position)
        self.old_velocity = deepcopy(self.velocity)

        position_2d = utils.coord_1d_to_2d(self.position)

        if self.qc_program == 'dftb+':

            self.dftbplus_instance.go_to_workdir()

            n_AO = sum( self.dftbplus_instance.worker.get_atom_nr_basis() )

            S = self.dftbplus_instance.worker.return_overlap_twogeom(position_2d, position_2d)

            self.dftbplus_instance.worker.set_geometry(position_2d)

            self.dftbplus_instance.worker.update_coordinate_dependent_stuffs()

            self.dftbplus_instance.return_from_workdir()

        elif self.qc_program == 'pyscf':
            
            position_2d = utils.coord_1d_to_2d(self.position)

            self.pyscf_instance.update_geometry(position_2d)

            S = self.pyscf_instance.get_overlap_matrix()

        else:
            
            utils.stop_with_error("Unknown quantum chemistry program %s .\n" % self.qc_program)

        n_spin = int(self.is_open_shell) + 1

        if self.S is not None:
            self.old_S = deepcopy(self.S)
        if self.Sinv is not None:
            self.old_Sinv = deepcopy(self.Sinv)
        if self.L is not None:
            self.old_L = deepcopy(self.L)

        self.S = utils.symmetrize(S, is_upper_triangle = True)
        self.t_S = self.t_position

        self.Sinv = np.linalg.inv(self.S)
        self.t_Sinv = self.t_S

        self.L = BasisTransformer.lowdin(self.S)

        if self.old_S is None:
            self.old_S = deepcopy(self.S)
        if self.old_Sinv is None:
            self.old_Sinv = deepcopy(self.Sinv)
        
        self.update_derivative_coupling()
        
        if self.do_interpol:
            self.reconstruct_gs(rho_H_only = True)

        return


    def update_mo_dependent_quantities(self):
        
        self.update_gs_density_matrix()

        self.update_gs_hamiltonian_and_molevels()

        self.update_estate_energies()

        self.update_mo_tdnac()

        self.update_tdnac()

        self.update_csc()

        return


    def update_e_coeffs_dependent_quantities(self):
        
        self.update_density_matrix()

        self.update_ehrenfest_energy()

        return


    def set_new_e_coeffs(self, e_coeffs, t_e_coeffs):
        
        self.old_e_coeffs = self.e_coeffs
        self.e_coeffs = e_coeffs
        self.t_e_coeffs = t_e_coeffs

        return


    def set_new_e_coeffs_tauderiv(self, e_coeffs_tauderiv, t_e_coeffs_tauderiv):
        
        self.old_e_coeffs_tauderiv = self.e_coeffs_tauderiv
        self.e_coeffs_tauderiv = e_coeffs_tauderiv
        self.t_e_coeffs_tauderiv = t_e_coeffs_tauderiv

        return


    def modify_gs_energy(self, dgs_energy, t_gs_energy):
        
        self.old_gs_energy = self.gs_energy
        self.gs_energy += dgs_energy
        self.t_gs_energy = t_gs_energy

        return


    def update_density_matrix(self):
        
        if self.basis == 'configuration' and self.excitation == 'cis' and not self.is_open_shell:

            utils.check_time_equal(self.t_mo_coeffs, self.t_gs_rho)
            utils.check_time_equal(self.t_mo_coeffs, self.t_e_coeffs)

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

            #print('CIS_COEFFS', cis_coeffs) ## Debug code

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
            self.t_rho = self.t_mo_coeffs

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

    
    def reconstruct_gs(self, is_initial = False, energy_only = False, rho_H_only = False):

        n_atom = len(self.atomparams)

        coords = self.position.reshape(n_atom, 3)

        mo_energies, mo_coeffs_real, gs_energy, gs_filling = self.get_scf_results(coords)
        
        self.gs_energy = gs_energy
        self.t_gs_energy = self.t_position

        if energy_only:
            return

        self.init_mo_energies = deepcopy(mo_energies)
        self.gs_filling = deepcopy(gs_filling)

        if rho_H_only:
        # DO NOT update self.mo_coeffs/mo_coeffs_nophase, but self.gs_rho will be updated according to the newly calculated MOs
            mo_coeffs = mo_coeffs_real.astype('complex128')
            mo_coeffs_nophase = mo_coeffs_real.astype('complex128')
        else:
            self.mo_coeffs = mo_coeffs_real.astype('complex128')
            self.mo_coeffs_nophase = mo_coeffs_real.astype('complex128')
            self.t_mo_coeffs = self.t_position
            self.t_mo_coeffs_nophase = self.t_position
            self.tau_mo_coeffs = self.t_position / self.alpha
            self.tau_mo_coeffs_nophase = self.t_position / self.alpha

        self.n_elec = np.sum(self.gs_filling)

        self.n_MO = np.size(self.mo_coeffs, 2)
        self.n_AO = self.n_MO
            
        if not rho_H_only:
            self.mo_e_int = np.zeros( (2, self.n_MO), dtype = 'float64')
            self.t_mo_e_int = self.t_position
            self.tau_mo_e_int = self.t_position / self.alpha

        self.update_gs_density_matrix()
        self.update_gs_hamiltonian_and_molevels(gs_hamiltonian_only = True)
            
        if is_initial:

            self.rho = self.gs_rho.astype('complex128')
            self.t_rho = self.t_position

            self.initial_gs_energy = self.gs_energy

            self.update_position_dependent_quantities()

            self.update_mo_dependent_quantities()

            self.update_e_coeffs_dependent_quantities()
                
        return


    def get_scf_results(self, coords):

        if self.qc_program == 'dftb+':
        
            self.dftbplus_instance.go_to_workdir()
            
            self.dftbplus_instance.worker.set_geometry(coords)

            gs_energy = self.dftbplus_instance.worker.get_energy()

            mo_energies, mo_coeffs_real = self.dftbplus_instance.worker.get_molecular_orbitals(
                open_shell = self.is_open_shell
            )

            gs_filling = self.dftbplus_instance.worker.get_filling(open_shell = self.is_open_shell)

            self.dftbplus_instance.return_from_workdir()

        elif self.qc_program == 'pyscf':
            
            self.pyscf_instance.update_geometry(coords)
            
            self.pyscf_instance.converge_scf()

            gs_energy = self.pyscf_instance.ks.e_tot

            if self.is_open_shell:
                utils.stop_with_error('Currently, pyscf-based implementation not compatible with open-shell systems.')

            mo_coeffs_real = np.array( [ self.pyscf_instance.ks.mo_coeff.transpose() ] )
            mo_energies = np.array( [ self.pyscf_instance.ks.mo_energy ] )

            gs_filling = np.array( [ self.pyscf_instance.ks.get_occ(mo_energies[0], mo_coeffs_real) ] )

        else:

            utils.stop_with_error("Not compatible with quantum chemistry program %s ." % self.qc_program)

        return mo_energies, mo_coeffs_real, gs_energy, gs_filling


    def non_orthogonality(self, update_cmo=True):

        n_atom = len(self.atomparams)

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        coords = self.position.reshape(n_atom, 3)
        
        mo_energies, mo_coeffs_real, gs_energy, gs_filling = self.get_scf_results(coords)

        if update_cmo:
            self.cmo_levels = deepcopy(mo_energies)
            self.cmo_coeffs = deepcopy(mo_coeffs_real)
            self.t_cmo = self.t_position

        n_occ = round( gs_filling.sum() / 2 )

        canonical_mo_coeffs = mo_coeffs_real.astype('complex128')

        for i_spin in range(n_spin):

            S = np.dot(
                np.dot(self.mo_coeffs_nophase[i_spin,:,:], self.S).conj(), np.transpose(canonical_mo_coeffs[i_spin,:,:])
            )

            S_occ = S[0:n_occ,0:n_occ]
            S_vir = S[n_occ:,n_occ:]

            det_S_occ = np.linalg.det(S_occ)
            det_S_vir = np.linalg.det(S_vir)

        return det_S_occ, det_S_vir


    def update_gs_density_matrix(self):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        if self.gs_rho is not None:
            self.old_gs_rho = deepcopy(self.gs_rho)
        
        self.gs_rho = np.zeros( (n_spin, self.n_AO, self.n_AO), dtype = 'float64' )

        # diagonal matrix whose elements are occupation numbers
        f = np.zeros( (self.n_MO, self.n_MO), dtype = 'complex128' )

        for i_spin in range(n_spin):

            scaled_mo_coeffs = np.zeros_like(self.mo_coeffs[i_spin,:,:])

            for i_MO in range(self.n_MO):
                
                scaled_mo_coeffs[i_MO,:] = self.gs_filling[i_spin][i_MO] * self.mo_coeffs[i_spin,i_MO,:]

            rho = np.dot( np.transpose(self.mo_coeffs[i_spin,:,:]).conjugate(), scaled_mo_coeffs )

            self.gs_rho[i_spin, :, :] = np.real(rho[:, :])
            self.t_gs_rho = self.t_mo_coeffs
        
        return
    

    def update_gs_hamiltonian_and_molevels(self, gs_hamiltonian_only = False):
        
        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        if self.H is not None:
            self.old_H = deepcopy(self.H)

        if self.qc_program == 'dftb+':

            self.dftbplus_instance.go_to_workdir()
            Htmp = self.dftbplus_instance.worker.return_hamiltonian(self.gs_rho)
            Htmp = self.dftbplus_instance.worker.return_hamiltonian(self.gs_rho) ## Calling twice here eliminates the spikes in MO levels; why?????
            self.dftbplus_instance.return_from_workdir()

        elif self.qc_program == 'pyscf':
            
            if not n_spin == 1:
                utils.stop_with_error("Currently not compatible with open-shell systems.")

            Htmp = self.pyscf_instance.return_hamiltonian(self.gs_rho[0,:,:], n_spin)

        self.H = np.zeros_like(Htmp, dtype = 'complex128')

        if gs_hamiltonian_only:

            for i_spin in range(n_spin):

                self.H[i_spin,:,:] = utils.hermitize(Htmp[i_spin,:,:], is_upper_triangle = True)
                #self.H[i_spin,:,:] = utils.hermitize(Htmp[i_spin,:,:], is_upper_triangle = False); print('DEBUG: LOWER TRIANGLE') ## Debug code

        else:

            self.mo_levels = np.zeros( (n_spin, self.n_MO), dtype = 'float64')

            for i_spin in range(n_spin):

                self.H[i_spin,:,:] = utils.hermitize(Htmp[i_spin,:,:], is_upper_triangle = True)

                H_MO = np.dot(
                    self.mo_coeffs[i_spin,:,:].conj(), np.dot(self.H[i_spin,:,:], self.mo_coeffs[i_spin,:,:].transpose())
                )

                #print('DIAG(H_MO)',np.diag(H_MO)) ## Debug code

                self.mo_levels[i_spin,:] = np.real(np.diag(H_MO))

        self.t_H = self.t_gs_rho
        
        if not gs_hamiltonian_only:
            utils.check_time_equal(self.t_H, self.t_mo_coeffs)
            self.t_mo_levels = self.t_mo_coeffs
        
        return


    def interpolate_matrices_and_nuclei(self, ibin_interpol, nbin_interpol, is_first_call = False, update_deriv_coupling = True):
        
        # interpolate H, S, and ground-state \rho between the old and the new nuclear steps
        #
        # self.H/S/gs_rho will be replaced with the interpolated ones
        # So, before that the "true" matrices of the new nuclear steps has to be saved in a new variable
        # (self.new_H, etc.)
        # This is done when is_first_call is True (usually, this is along with ibin_interpol = 1)

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1

        if is_first_call:
            
            self.new_L = BasisTransformer.lowdin(self.S)
            
            self.new_H = deepcopy(self.H)
            self.new_gs_rho = deepcopy(self.gs_rho)
            self.new_deriv_coupling = deepcopy(self.deriv_coupling)

            self.new_position = deepcopy(self.position)
            self.new_velocity = deepcopy(self.velocity)
        
        B = float(ibin_interpol) / float(nbin_interpol)
        A = 1.0 - B

        self.position = A * self.old_position + B * self.new_position
        self.velocity = A * self.old_velocity + B * self.new_velocity

        position_2d = utils.coord_1d_to_2d(self.position)

        if self.qc_program == 'dftb+':
        
            # Update internal coordinate in DFTB+

            self.dftbplus_instance.worker.set_geometry(position_2d)

            self.dftbplus_instance.worker.update_coordinate_dependent_stuffs()

            # Exact S in the interpolated position
            
            S = self.dftbplus_instance.worker.return_overlap_twogeom(position_2d, position_2d)

        elif self.qc_program == 'pyscf':
            
            self.pyscf_instance.update_geometry(position_2d)

            S = self.pyscf_instance.mol.intor('int1e_ovlp')

        else:

            utils.stop_with_error("Unknown QC program %s ." % self.qc_program)

        self.S = utils.symmetrize(S, is_upper_triangle = True)
        self.Sinv = np.linalg.inv(self.S)
        
        # Set lowdin orthogonalized basis at the current (interpolated) position

        self.L = BasisTransformer.lowdin(self.S)

        # Interpolation in the Lowdin basis

        for i_spin in range(n_spin):

            self.H[i_spin,:,:] = self.interpolate_in_orthogonal_basis(self.old_H[i_spin,:,:], self.new_H[i_spin,:,:], A)
            self.gs_rho[i_spin,:,:] = self.interpolate_in_orthogonal_basis(self.old_gs_rho[i_spin,:,:], self.new_gs_rho[i_spin,:,:], A)
            
        if update_deriv_coupling or self.new_deriv_coupling is None or self.old_deriv_coupling is None:
            self.update_derivative_coupling()
        else:
            self.deriv_coupling = self.interpolate_in_orthogonal_basis(self.old_deriv_coupling, self.new_deriv_coupling, A)

        return


    def interpolate_in_orthogonal_basis(self, old_M, new_M, A):

        old_M_in_L = BasisTransformer.ao2mo(old_M, self.old_L)
        new_M_in_L = BasisTransformer.ao2mo(new_M, self.new_L)

        M_in_L = A * old_M_in_L + (1.0 - A) * new_M_in_L

        M_in_AO = BasisTransformer.mo2ao(M_in_L, self.S, self.L)

        return M_in_AO


    def set_mo_overlap_with_old_other_e_part(self, old_e_part):

        if self.is_open_shell:
            n_spin = 2
        else:
            n_spin = 1
        
        position_2d_old = utils.coord_1d_to_2d(old_e_part.old_position)
        position_2d_new = utils.coord_1d_to_2d(self.position)

        if self.qc_program == 'dftb+':

            # DFTB+

            self.dftbplus_instance.go_to_workdir()

            n_AO = sum( self.dftbplus_instance.worker.get_atom_nr_basis() )

            S = self.dftbplus_instance.worker.return_overlap_twogeom(position_2d_old, position_2d_new)

            self.dftbplus_instance.return_from_workdir()

            S = utils.symmetrize(S, is_upper_triangle = True)

            # End DFTB+

        else:

            utils.stop_with_error("This function is currently not compatible with QC program %s ." % self.qc_program)

        self.mo_overlap_with_old_other = np.zeros_like(self.mo_coeffs)

        for i_spin in range(n_spin):

            self.mo_overlap_with_old_other = np.dot(
                old_e_part.old_mo_coeffs[i_spin,:,:], np.dot(
                    S, self.mo_coeffs[i_spin,:,:]
                )
            )

        return


    def get_phase_cancellation_angle(self, i_spin, i_mo):
        
        # multiply e^i\theta to MO coeffs to minimize square-sum of imaginary part
        # find \theta
        
        c = self.mo_coeffs[i_spin,i_mo,:]

        c_real = np.real(c)
        c_imag = np.imag(c)

        # find \theta such that d(s2-norm of imaginary part)/d\theta = 0
        # theta minimize OR maximize the s2-norm; compare two results and return the \theta MINIMIZE the s2-norm

        val = 2.0 * np.dot(c_real,c_imag) / ( np.linalg.norm(c_imag)**2 - np.linalg.norm(c_real)**2 )

        theta1 = 0.5 * np.arctan(val)
        theta2 = theta1 + 0.5 * np.pi

        phase1 = np.exp( (0.0+1.0j) * theta1 )
        phase2 = np.exp( (0.0+1.0j) * theta2 )

        res1 = np.linalg.norm( np.imag( phase1 * c ) )
        res2 = np.linalg.norm( np.imag( phase2 * c ) )

        if res1 < res2:
            theta = theta1
            res = res1
        else:
            theta = theta2
            res = res2
        
        return theta, res


    def get_overlap_with_old_other_e_part(self, i_state_old, i_state_new):

        # !!! set_mo_overlap_with_old_other_epart must be called in advance !!!

        # excitation = 'cis' and  basis type = 'configuration'

        if self.is_open_shell:
            utils.stop_with_error('Electronic-state overlap calculations currently not compatible with open-shell systems')
        
        n_act_occ = len(self.active_occ_mos)
        n_act_vir = len(self.active_vir_mos)

        i_act_old = i_state_old // n_act_occ
        a_act_old = i_state_old %  n_act_occ
        i_act_new = i_state_new // n_act_occ
        a_act_new = i_state_new %  n_act_occ

        i_old = self.active_occ_mos[i_act_old]
        a_old = self.active_occ_mos[a_act_old]
        i_new = self.active_occ_mos[i_act_new]
        a_new = self.active_occ_mos[a_act_new]

        n_occ = self.n_occ

        S = self.mo_overlap_with_old_other[0,0:n_occ,0:n_occ]

        # overlap = det(S[ia,jb]) * det(S[ii,jj]) + det(S[ia,jj]) * det(S[ii,jb])

        S_iijj = np.linalg.det(S)

        S[0,i_old,0:n_occ] = self.mo_overlap_with_old_other[0,a_old,0:n_occ]

        S_iajj = np.linalg.det(S)

        S[0,0:n_occ,i_new] = self.mo_overlap_with_old_other[0,0:n_occ,a_new]
        S[0,i_old,i_new] = self.mo_overlap_with_old_other[0,a_old,a_new]

        S_iajb = np.linalg.det(S)

        S = self.mo_overlap_with_old_other[0,0:n_occ,0:n_occ]
        S[0,0:n_occ,i_new] = self.mo_overlap_with_old_other[0,0:n_occ,a_new]
        
        S_iijb = np.linalg.det(S)

        overlap = S_iajb * S_iijj + S_iajj * S_iijb

        return overlap

                
    def dump_matrices(self):
        
        matrices = {
            'gs_rho'              : deepcopy(self.gs_rho),
            'rho'                 : deepcopy(self.rho),
            'H'                   : deepcopy(self.H),
            'S'                   : deepcopy(self.S),
            'Sinv'                : deepcopy(self.Sinv),
            'tdnac'               : deepcopy(self.tdnac),
            'deriv_coupling'      : deepcopy(self.deriv_coupling),
            'mo_coeffs'           : deepcopy(self.mo_coeffs),
            'mo_coeffs_nophase'   : deepcopy(self.mo_coeffs_nophase),
            'old_mo_coeffs'       : deepcopy(self.old_mo_coeffs),
            'old_mo_coeffs_nophase': deepcopy(self.old_mo_coeffs_nophase),
            'mo_e_int'            : deepcopy(self.mo_e_int),
            'mo_levels'           : deepcopy(self.mo_levels),
            'init_mo_energies'    : deepcopy(self.init_mo_energies),
            'estate_energies'     : deepcopy(self.estate_energies),
            'old_estate_energies' : deepcopy(self.old_estate_energies),
            'deriv_coupling'      : deepcopy(self.deriv_coupling),
            'gs_energy'           : self.gs_energy,
            'initial_gs_energy'   : self.initial_gs_energy,
            'gs_filling'          : deepcopy(self.gs_filling),
            'n_MO'                : self.n_MO,
            'n_AO'                : self.n_AO,
            'n_elec'              : self.n_elec,
        }

        return matrices
