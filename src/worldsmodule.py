#-*- coding: utf-8 -*-

import numpy as np
from tbfmodule import Tbf
from constants import ATOM_MASSES_AMU, DEFAULT_GWP_WIDTH_AU, H_DIRAC, AMU2AU, AU2SEC
from copy import deepcopy
from utils import stop_with_error, Timer
from settingsmodule import load_setting
from integratormodule import Integrator
from basistransformermodule import BasisTransformer
from files import GlobalOutputFiles

class World:
    
    worlds = []


    def __init__(self, settings):
        
        self.settings = deepcopy(settings)

        self.live_tbf_count  = 0

        self.tbfs                  = []
        self.tbf_coeffs            = np.array([], dtype='complex128')
        self.tbf_coeffs_nophase    = np.array([], dtype='complex128')
        self.e_int                 = np.array([], dtype='complex128') # phase factor = exp(-i/\hbar * e_int)
        #self.tbf_coeffs_tderiv     = np.zeros_like(self.tbf_coeffs)
        self.old_tbf_coeffs        = None
        #self.old_tbf_coeffs_tderiv = None

        self.S_tbf = None
        self.H_tbf = None
        self.L_tbf = None

        self.H_tbf_diag_origin = None

        self.world_id = len(World.worlds)

        self.integmethod = load_setting(settings, 'integrator')

        self.tbf_coeffs_nophase_integrator = Integrator(self.integmethod, mode = 'chasing')
        self.e_int_integrator              = Integrator(self.integmethod, mode = 'chasing')

        self.tbf_coeffs_are_trivial = load_setting(settings, 'tbf_coeffs_are_trivial')

        self.print_xyz_interval = load_setting(settings, 'print_xyz_interval')

        self.flush_interval = load_setting(settings, 'flush_interval')

        self.cloning_rule = load_setting(settings, 'cloning_rule')
        self.cloning_parameters = load_setting(settings, 'cloning_parameters')

        self.max_n_tbf = load_setting(settings, 'max_n_tbf')

        self.do_interpol = load_setting(settings, 'do_tbf_interpol')

        if self.do_interpol:
            self.nbin_interpol = load_setting(settings, 'nbin_interpol_tbf')
        else:
            self.nbin_interpol = 1
        
        self.dtau = load_setting(settings, 'dtau')
        self.alpha = load_setting(settings, 'alpha')
        self.dt = self.alpha * self.dtau # in interpolation, dt = self.dt / nbin

        self.globaloutput = GlobalOutputFiles(self.world_id)

        self.istep = 0

        World.worlds.append(self)
        
        return


    def set_initial_state(self, atomparams, position, velocity, is_fixed = False):

        if len(self.tbfs) > 0:

            stop_with_error("Error: world must be empty (no existing TBF) when calling set_initial_position_velocity.")
        
        self.atomparams = deepcopy(atomparams)

        natom = len(self.atomparams)
        n_dof = 3 * natom

        mass  = [ 0.0  for i in range(n_dof) ]
        width = [ 0.0  for i in range(n_dof) ]
        elems = [ None for i in range(n_dof) ]

        for iatom, atomparam in enumerate(atomparams):

            mass_atom  = ATOM_MASSES_AMU[atomparam.elem]
            width_atom = DEFAULT_GWP_WIDTH_AU[atomparam.elem]
            elem_atom  = atomparam.elem

            mass[ 3*iatom:3*iatom+3] = [mass_atom , mass_atom , mass_atom ]
            width[3*iatom:3*iatom+3] = [width_atom, width_atom, width_atom]
            elems[3*iatom:3*iatom+3] = [elem_atom , elem_atom , elem_atom ]

        mass  = np.array(mass)
        width = np.array(width)

        momentum = mass * AMU2AU * velocity # be careful for unit of mass

        initial_estates = load_setting(self.settings, 'initial_estates')
        initial_coeffs  = load_setting(self.settings, 'initial_coeffs')
        
        # normalize initial TBF coefficients
        initial_coeffs = np.array(initial_coeffs)
        initial_coeffs /= np.linalg.norm(initial_coeffs)

        n_coeff = len(initial_coeffs)
        if n_coeff != len(initial_estates):
            utils.stop_with_error('Number of initial estates and coeffs must be the same.\n')

        initial_e_coeffs_nophase = [ 0.0+0.0j for i in range(self.settings['n_estate']) ]
        initial_e_coeffs_e_int   = [ 0.0+0.0j for i in range(self.settings['n_estate']) ]

        for i_coeff in range(n_coeff):

            initial_e_coeffs_nophase[initial_estates[i_coeff]] = 1.0+0.0j

        initial_e_coeffs_nophase = np.array(initial_e_coeffs_nophase) 
        initial_e_coeffs_e_int   = np.array(initial_e_coeffs_e_int) 

        initial_e_coeffs_nophase /= np.linalg.norm(initial_e_coeffs_nophase) # normalize

        initial_tbf = Tbf(
            self.settings, self.atomparams, position, n_dof,
            self.settings['n_estate'], len(self.tbfs),
            momentum = momentum, mass = mass, width = width,
            e_coeffs_nophase = initial_e_coeffs_nophase,
            e_coeffs_e_int = initial_e_coeffs_e_int,
            is_fixed = is_fixed,
        )

        self.add_tbf(initial_tbf, coeff_nophase = 1.0+0.0j, normalize = False)

        self.construct_tbf_hamiltonian()

        return


    def add_tbf(self, tbf, coeff_nophase=0.0+0.0j, e_int=0.0+0.0j, normalize = True):

        self.tbfs.append(tbf)

        coeff = coeff_nophase * np.exp( (-1.0j/H_DIRAC) * e_int )

        self.tbf_coeffs_nophase = np.append(self.tbf_coeffs_nophase, coeff_nophase)
        self.tbf_coeffs         = np.append(self.tbf_coeffs, coeff)
        self.e_int              = np.append(self.e_int, e_int)

        if self.old_tbf_coeffs is not None:
            self.old_tbf_coeffs    = np.append(self.old_tbf_coeffs, coeff)

        if normalize:
            
            self.tbf_coeffs_nophase /= np.linalg.norm(self.tbf_coeffs_nophase)
            self.tbf_coeffs /= np.linalg.norm(self.tbf_coeffs)
            self.old_tbf_coeffs /= np.linalg.norm(self.old_tbf_coeffs) # OK?

        return

    
    def propagate(self):

        # ordering OK?

        self.print_results()

        self.update_position_and_velocity()
        self.update_electronic_part()
        self.update_nuclear_part()

        self.istep += 1

        return
    

    def destroy_tbf(self, tbf):
        
        self.live_tbf_count -= 1

        return


    def update_position_and_velocity(self):
        
        for guy in self.tbfs:

            if not guy.is_alive:
                continue
            
            guy.update_position_and_velocity(self.dt)

        return
    

    def update_electronic_part(self):

        for guy in self.tbfs: # 'guy' is an individual tbf
            
            if not guy.is_alive:
                continue

            guy.update_electronic_part(self.dt)

        return


    def make_tbf_coeffs_nophase_tderiv(self, t, tbf_coeffs_nophase):

        n_tbf = self.get_total_tbf_count()

        H_tbf_nophase = deepcopy(self.H_tbf)

        for i_tbf in range(n_tbf):
            H_tbf_nophase[i_tbf, i_tbf] = 0.0+0.0j

        tbf_coeffs_nophase_tderiv = (-1.0j / H_DIRAC) * np.dot(
            np.linalg.inv(self.S_tbf), np.dot(H_tbf_nophase, tbf_coeffs_nophase.transpose())
        ).transpose()

        return tbf_coeffs_nophase_tderiv


    def make_e_int_tderiv(self, t, e_int):
        
        n_tbf = self.get_total_tbf_count()

        e_int_tderiv = np.zeros_like(e_int)

        for i_tbf in range(n_tbf):

            e_int_tderiv[i_tbf] = self.H_tbf[i_tbf,i_tbf]

        return e_int_tderiv


    def cloning(self, i_tbf):
        
        tbf = self.tbfs[i_tbf]

        clone = False

        if self.cloning_rule == 'nstate': # placeholder
            
            utils.stop_with_error("Cloning rule nstate under construction.") ## Debug code

            e_coeffs = tbf.e_part.get_e_coeffs()

            eff_n_states = 1.0 / np.sum( np.abs(e_coeffs)**4 )

        elif self.cloning_rule == 'energy':
            
            ehrenfest_e = tbf.e_part.get_ehrenfest_energy()

            e_coeffs = tbf.e_part.get_e_coeffs()

            e_popul = np.abs(e_coeffs)**2

            e_popul_sorted = np.sort(e_popul)[::-1]
            e_index_sorted = np.argsort(e_popul)[::-1]

            n_estate = len(e_coeffs)

            estate_energies = tbf.e_part.get_estate_energies()

            for k_estate in range(1, n_estate): # order of population
                # k_estate == 0 is skipped; this is 'main' state.
                # Only 'sub' state should be transffered to the spawned TBF

                i_estate = e_index_sorted[k_estate]

                # Modified Ehrenfest energy if the contribution of state i is set to zero

                ehrenfest_e_without_i = 0.0

                e_popul_without_i = deepcopy(e_popul)
                e_popul_without_i[i_estate] = 0.0
                e_popul_without_i /= ( np.sum(e_popul_without_i) / np.sum(e_popul) ) # sum of e_popul may differ from unity because of numerical errors

                for j_estate in range(0, n_estate):

                    ehrenfest_e_without_i += e_popul_without_i[j_estate] * estate_energies[j_estate]

                delta_e = ehrenfest_e_without_i - ehrenfest_e

                if abs(delta_e) > self.cloning_parameters['e_threshold']:
                    
                    if clone:
                        utils.stop_with_error('Current implementation cannot clone 2 or more TBFs at once.')

                    clone = True
                    clone_state = i_estate
        else:

            utils.stop_with_error(
                "Unknown cloning rule %s ." % self.cloning_rule
            )


        if clone:

            print('CLONE')

            e_coeff_i = tbf.e_part.e_coeffs[clone_state]
            e_coeff_nophase_i = tbf.e_coeffs_nophase[clone_state]
            e_e_int_i = tbf.e_coeffs_e_int[clone_state]
            
            baby = tbf.spawn( clone_state, len(self.tbfs) )
            
            #tbf.e_part.deliver(clone_state) ## Modification of e coeffs

            self.tbfs.append(baby)

            c_i = self.tbf_coeffs[i_tbf]
            c_nophase_i = self.tbf_coeffs_nophase[i_tbf]

            self.tbf_coeffs_nophase = np.append(
                self.tbf_coeffs_nophase, self.tbf_coeffs_nophase[i_tbf]*abs(e_coeff_i)
            )
            self.tbf_coeffs_nophase[i_tbf] = self.tbf_coeffs_nophase[i_tbf] * np.sqrt(1.0 - abs(e_coeff_i)**2)

            self.tbf_coeffs = np.append(
                self.tbf_coeffs, self.tbf_coeffs[i_tbf]*e_coeff_i
            )
            self.tbf_coeffs[i_tbf] = self.tbf_coeffs[i_tbf] * np.sqrt(1.0 - abs(e_coeff_i)**2)

            self.e_int = np.append(self.e_int, e_e_int_i)

            self.tbf_coeffs_nophase_integrator.reset()
            self.e_int_integrator.reset()

            n_tbf = self.get_total_tbf_count()
            
            #self.update_nuclear_part()

        return


    def initialize_tbf_coeffs_integrator(self):

        self.construct_tbf_hamiltonian()

        tbf_coeffs_nophase = self.get_tbf_coeffs_nophase()
        
        self.tbf_coeffs_nophase_integrator.initialize_history(
            0.0, self.tbf_coeffs_nophase, self.make_tbf_coeffs_nophase_tderiv,
        )

        self.e_int_integrator.initialize_history(
            0.0, self.e_int, self.make_e_int_tderiv,
        )

        return


    def update_tbf_overlap(self):

        n_tbf = self.get_total_tbf_count()

        if self.S_tbf is not None:
            self.old_S_tbf = deepcopy(self.S_tbf)
        if self.L_tbf is not None:
            self.old_L_tbf = deepcopy(self.L_tbf)

        self.S_tbf = np.zeros( (n_tbf, n_tbf), dtype = 'complex128' )

        for i_tbf in range(n_tbf):
            
            guy_i = self.tbfs[i_tbf]

            if not guy_i.is_alive:
                continue

            for j_tbf in range(i_tbf,n_tbf):

                guy_j = self.tbfs[j_tbf]

                if not guy_j.is_alive:
                    continue

                g_ij = Tbf.get_gaussian_overlap(guy_i, guy_j)

                S_ij = Tbf.get_wf_overlap(guy_i, guy_j, gaussian_overlap = g_ij)
                
                self.S_tbf[i_tbf,j_tbf] = S_ij
                self.S_tbf[j_tbf,i_tbf] = np.conjugate(S_ij)

        # lowdin orghogonalization

        self.L_tbf = BasisTransformer.lowdin(self.S_tbf)

        return


    def construct_tbf_hamiltonian(self):

        # update overlap
        self.update_tbf_overlap()

        # construct TBF Hamiltonian

        n_tbf = self.get_total_tbf_count()

        #if self.S_tbf is not None:
        #    self.old_S_tbf = deepcopy(self.S_tbf)
        if self.H_tbf is not None:
            self.old_H_tbf = deepcopy(self.H_tbf)
        #if self.L_tbf is not None:
        #    self.old_L_tbf = deepcopy(self.L_tbf)

        #print('S_tbf', self.S_tbf) ## Debug code

        #self.S_tbf = np.zeros( (n_tbf, n_tbf), dtype = 'complex128' )
        self.H_tbf = np.zeros( (n_tbf, n_tbf), dtype = 'complex128' )

        for i_tbf in range(n_tbf):
            
            guy_i = self.tbfs[i_tbf]

            if not guy_i.is_alive:
                continue

            #for j_tbf in range(n_tbf):
            for j_tbf in range(i_tbf,n_tbf):

                guy_j = self.tbfs[j_tbf]

                if not guy_j.is_alive:
                    continue

                g_ij = Tbf.get_gaussian_overlap(guy_i, guy_j)

                #S_ij = Tbf.get_wf_overlap(guy_i, guy_j, gaussian_overlap = g_ij)
                H_ij = Tbf.get_tbf_hamiltonian_element_BAT(guy_i, guy_j, gaussian_overlap = g_ij)
                #H_ij = 0.0 ## Debug code
                
                #self.S_tbf[i_tbf,j_tbf] = S_ij
                self.H_tbf[i_tbf,j_tbf] = H_ij

                #self.S_tbf[j_tbf,i_tbf] = S_ij
                self.H_tbf[j_tbf,i_tbf] = np.conj(H_ij)

        #print('H_tbf', self.H_tbf) ## Debug code

        # < \psi_m | d/dt | \psi_n >

        for i_tbf in range(n_tbf):
            
            guy_i = self.tbfs[i_tbf]

            if not guy_i.is_alive:
                continue

            #for j_tbf in range(n_tbf):
            for j_tbf in range(i_tbf,n_tbf):

                guy_j = self.tbfs[j_tbf]

                if not guy_j.is_alive:
                    continue

                g_ij = Tbf.get_gaussian_overlap(guy_i, guy_j)

                val = Tbf.get_tbf_derivative_coupling(guy_i, guy_j, g_ij)
                #val = 0.0; print('DEBUG: TBF DERIV COUPLING SET TO ZERO') ## Debug code

                #if i_tbf == j_tbf:
                #    val = 0.0 # ?

                self.H_tbf[i_tbf,j_tbf] -= 1.0j * H_DIRAC * val
                self.H_tbf[j_tbf,i_tbf] -= 1.0j * H_DIRAC * val

        #print('H_tbf+\Sigma_tbf', self.H_tbf) ## Debug code

        ## symmetrize S

        #for i in range(n_tbf):
        #    for j in range(i, n_tbf):

        #        val = 0.5 * (self.S_tbf[i,j] + self.S_tbf[j,i])
        #        self.S_tbf[i,j] = val
        #        self.S_tbf[j,i] = val

        #        #val = 0.5 * ( self.H_tbf[i,j] + np.conjugate(self.H_tbf[j,i]) )
        #        #self.H_tbf[i,j] = val
        #        #self.H_tbf[j,i] = np.conjugate(val)

        ## lowdin orghogonalization

        #self.L = BasisTransformer.lowdin(self.S_tbf)

        # subtract energy origin

        if self.H_tbf_diag_origin is None:
            self.H_tbf_diag_origin = np.diag(self.H_tbf)[0]
        
        for i in range(n_tbf):
            self.H_tbf[i,i] -= self.H_tbf_diag_origin

        return


    def update_nuclear_part(self):

        # cloning if necessary

        n_tbf = self.get_total_tbf_count()

        if n_tbf < self.max_n_tbf or self.max_n_tbf < 0:

            for i_tbf in range(n_tbf):

                self.cloning(i_tbf)

        # (in the initial step) initialize integrator
        if not self.tbf_coeffs_nophase_integrator.is_history_initialized:
            self.initialize_tbf_coeffs_integrator()

        dt = self.dt / float(self.nbin_interpol)

        # construct TBF Hamiltonian

        self.construct_tbf_hamiltonian()

        # Time propagation on interpolation

        for ibin_interpol in range(1, self.nbin_interpol+1):

            if ibin_interpol == 1:
                is_first_call = True
            else:
                is_first_call = False

            self.interpolate_matrices_and_nuclei(ibin_interpol, self.nbin_interpol, is_first_call)            

            # propagation of TBF coeffs
            
            tbf_coeffs_nophase = self.get_tbf_coeffs_nophase()

            if n_tbf == 1 and self.tbf_coeffs_are_trivial:

                new_tbf_coeffs = np.array( [ 1.0+0.0j ] )
                new_tbf_coeffs_nophase = deepcopy(new_tbf_coeffs)
            
            else:

                new_tbf_coeffs_nophase = self.tbf_coeffs_nophase_integrator.engine(
                    dt, 0.0, tbf_coeffs_nophase, self.make_tbf_coeffs_nophase_tderiv,
                )
                
                new_e_int = self.e_int_integrator.engine(
                    dt, 0.0, self.e_int, self.make_e_int_tderiv,
                )

                self.e_int = new_e_int

                phase_factor = np.exp( (-1.0j/H_DIRAC) * new_e_int )

                new_tbf_coeffs = new_tbf_coeffs_nophase * phase_factor

            self.set_new_tbf_coeffs(new_tbf_coeffs)
            self.set_new_tbf_coeffs_nophase(new_tbf_coeffs_nophase)

            #print('H_tbf', self.H_tbf) ## Debug code
            #print('S_tbf', self.S_tbf) ## Debug code
            #print('TBF_COEFFS_NOPHASE', self.tbf_coeffs_nophase) ## Debug code
            #print('TBF_COEFFS', self.tbf_coeffs) ## Debug code
            #print('E_COEFFS_NOPHASE1', self.tbfs[0].e_coeffs_nophase) ## Debug code

        return


    def interpolate_matrices_and_nuclei(self, ibin_interpol, nbin_interpol, is_first_call = False):

        n_tbf = self.get_total_tbf_count()
        
        if is_first_call:
            
            self.new_L_tbf = BasisTransformer.lowdin(self.S_tbf)
            self.new_S_tbf = deepcopy(self.S_tbf)

            self.new_H_tbf = deepcopy(self.H_tbf)

            self.old_positions = []
            self.new_positions = []
            self.old_momenta = []
            self.new_momenta = []

            for i_tbf in range(n_tbf):
                
                tbf = self.tbfs[i_tbf]

                self.old_positions.append( deepcopy(tbf.old_position) )
                self.new_positions.append( deepcopy(tbf.position) )
                self.old_momenta.append( deepcopy(tbf.old_momentum) )
                self.new_momenta.append( deepcopy(tbf.momentum) )

            self.old_positions = np.array(self.old_positions)
            self.new_positions = np.array(self.new_positions)
            self.old_momenta = np.array(self.old_momenta)
            self.new_momenta = np.array(self.new_momenta)

        B = float(ibin_interpol) / float(nbin_interpol)
        A = 1.0 - B

        #self.positions = A * self.old_positions + B * self.new_positions
        #self.momenta   = A * self.old_momenta   + B * self.new_momenta

        ## set interpolated positions and momenta to TBFs (to update S_tbf)

        #for i_tbf in range(n_tbf):

        #    tbf = self.tbfs[i_tbf]

        #    tbf.position = self.positions[i_tbf]
        #    tbf.momenta  = self.momenta[i_tbf]

        ## update S_tbf

        #self.update_tbf_overlap()

        ## reset interpolated positions and momenta of TBFs

        #for i_tbf in range(n_tbf):

        #    tbf = self.tbfs[i_tbf]

        #    tbf.position = self.new_positions[i_tbf]
        #    tbf.momenta  = self.new_momenta[i_tbf]

        #print('S_tbf', self.S_tbf) ## Debug code

        self.S_tbf = A * self.old_S_tbf + B * self.new_S_tbf
        self.L_tbf = BasisTransformer.lowdin(self.S_tbf)

        # interpolate H_tbf

        self.H_tbf = self.interpolate_in_orthonormal_basis(self.old_H_tbf, self.new_H_tbf, A)

        return


    def interpolate_in_orthonormal_basis(self, old_M, new_M, A):
        
        old_M_in_L = BasisTransformer.ao2mo(old_M, self.old_L_tbf)
        new_M_in_L = BasisTransformer.ao2mo(new_M, self.new_L_tbf)

        M_in_L = A * old_M_in_L + (1.0 - A) * new_M_in_L

        M_in_TBF = BasisTransformer.mo2ao(M_in_L, self.S_tbf, self.L_tbf)

        return M_in_TBF
    

    def get_total_tbf_count(self):
        #return self.total_tbf_count
        return len(self.tbfs)


    def get_live_tbf_count(self):
        return self.live_tbf_count


    def get_tbf_coeffs(self):
        return deepcopy(self.tbf_coeffs)


    def get_tbf_coeffs_nophase(self):
        return deepcopy(self.tbf_coeffs_nophase)


    #def get_tbf_coeffs_tderiv(self):
    #    return deepcopy(self.tbf_coeffs_tderiv)


    def get_old_tbf_coeffs(self):
        return deepcopy(self.old_tbf_coeffs)


    def set_new_tbf_coeffs(self, new_tbf_coeffs):
        
        self.old_tbf_coeffs = deepcopy(self.tbf_coeffs)
        self.tbf_coeffs     = new_tbf_coeffs

        return


    def set_new_tbf_coeffs_nophase(self, new_tbf_coeffs_nophase):
        
        self.old_tbf_coeffs_nophase = deepcopy(self.tbf_coeffs_nophase)
        self.tbf_coeffs_nophase     = new_tbf_coeffs_nophase

        return


    #def set_new_tbf_coeffs_tderiv(self, new_tbf_coeffs_tderiv):
    #    
    #    self.old_tbf_coeffs_tderiv = deepcopy(self.tbf_coeffs_tderiv)
    #    self.tbf_coeffs_tderiv     = new_tbf_coeffs_tderiv

    #    return


    def print_results(self):

        if self.print_xyz_interval != 0 and (self.istep % self.print_xyz_interval) == 0:

            self.print_tbf_coeff_and_popul()

            self.print_timing_info()

        if self.istep > 0 and (self.istep % self.flush_interval) == 0:

            self.globaloutput.flush()

        return


    def print_tbf_coeff_and_popul(self):

        tbf_coeff_file = self.globaloutput.tbf_coeff
        tbf_coeff_nophase_file = self.globaloutput.tbf_coeff_nophase
        tbf_popul_file = self.globaloutput.tbf_popul

        t = self.dt * self.istep
        
        time_str = "STEP %12d T= %20.12f fs" % (self.istep, t*AU2SEC*1.0e15)
        
        tbf_coeff_file.write(time_str)
        tbf_coeff_nophase_file.write(time_str)
        tbf_popul_file.write(time_str)

        n_tbf = self.get_total_tbf_count()
        
        for i_tbf in range(n_tbf):

            tbf_coeff_file.write( " %20.12f+%20.12fj" % ( self.tbf_coeffs[i_tbf].real, self.tbf_coeffs[i_tbf].imag ) )
            tbf_coeff_nophase_file.write( " %20.12f+%20.12fj" % ( self.tbf_coeffs_nophase[i_tbf].real, self.tbf_coeffs_nophase[i_tbf].imag ) )
            tbf_popul_file.write( " %20.12f" % ( abs(self.tbf_coeffs[i_tbf])**2 ) )

        tbf_coeff_file.write("\n")
        tbf_coeff_nophase_file.write("\n")
        tbf_popul_file.write("\n")
        
        return


    def print_timing_info(self):

        lap_time = Timer.set_checkpoint_time(
            'world_print', return_laptime = True
        )
        
        elapsed_time = Timer.get_checkpoint_time('world_print') - Timer.get_checkpoint_time('program_start')

        time_file = self.globaloutput.time

        t = self.dt * self.istep
        
        time_str = "STEP %12d T= %20.12f fs" % (self.istep, t*AU2SEC*1.0e15)
        
        time_file.write(time_str)

        time_file.write( " %20.6f sec %20.6f sec" % (elapsed_time, lap_time) )

        time_file.write("\n")

        return
