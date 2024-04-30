#-*- coding: utf-8 -*-

import numpy as np
from tbfmodule import Tbf
from constants import ATOM_MASSES_AMU, DEFAULT_GWP_WIDTH_AU, H_DIRAC, AMU2AU
from copy import deepcopy
from utils import stop_with_error
from settingsmodule import load_setting
from integratormodule import Integrator

class World:
    
    worlds = []


    def __init__(self, settings):
        
        self.settings = deepcopy(settings)

        #self.total_tbf_count = 0
        self.live_tbf_count  = 0

        self.tbfs                  = []
        self.tbf_coeffs            = np.array([], dtype='complex128')
        self.tbf_coeffs_tderiv     = np.zeros_like(self.tbf_coeffs)
        #self.old_tbf_coeffs        = np.zeros_like(self.tbf_coeffs)
        #self.old_tbf_coeffs_tderiv = np.zeros_like(self.tbf_coeffs)
        self.old_tbf_coeffs        = None
        self.old_tbf_coeffs_tderiv = None

        self.dt = settings['dt']

        self.S_tbf = None
        self.H_tbf = None

        self.H_tbf_diag_origin = None

        self.world_id = len(World.worlds)

        #self.integmethod = load_setting(settings, 'integrator')

        #self.integrator = Integrator(self.integmethod)
        self.integrator = Integrator('adams_moulton_2')
        #print('WARNING: TBF coeffs integrated with leapfrog method.') ## Debug code
        #self.integrator = Integrator('leapfrog') ## Debug code

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

        n_tbf = len(initial_coeffs)
        if n_tbf != len(initial_estates):
            utils.stop_with_error('Number of initial estates and coeffs must be the same.\n')

        for i_tbf in range(n_tbf):

            initial_e_coeffs = [ 0.0+0.0j for i in range(self.settings['n_estate']) ]
            initial_e_coeffs[initial_estates[i_tbf]] = 1.0+0.0j
            ### Debug code
            #print('WARNING: INITIAL STATE NOT CORRECT')
            #initial_e_coeffs[1] = 1.0
            #initial_e_coeffs[2] = 1.0
            #initial_e_coeffs /= np.linalg.norm(initial_e_coeffs)
            ### End Debug code
            initial_e_coeffs = np.array(initial_e_coeffs)

            initial_tbf = Tbf(
                self.settings, self.atomparams, position, n_dof, self.settings['n_estate'], len(self.tbfs),
                momentum = momentum, mass = mass, width = width, e_coeffs = initial_e_coeffs,
                is_fixed = is_fixed,
            )

            self.add_tbf(initial_tbf, coeff = initial_coeffs[i_tbf], normalize = False)

        return


    def add_tbf(self, tbf, coeff = 0.0+0.0j, normalize = True):

        self.tbfs.append(tbf)

        self.tbf_coeffs        = np.append(self.tbf_coeffs, coeff)
        self.tbf_coeffs_tderiv = np.append(self.tbf_coeffs_tderiv, 0.0+0.0j)
        if self.old_tbf_coeffs is not None:
            self.old_tbf_coeffs    = np.append(self.old_tbf_coeffs, coeff)
            self.old_tbf_coeffs_tderiv = np.append(self.old_tbf_coeffs_tderiv, 0.0+0.0j)

        if normalize:

            self.tbf_coeffs /= np.linalg.norm(self.tbf_coeffs)
            self.old_tbf_coeffs /= np.linalg.norm(self.old_tbf_coeffs) # OK?

        return

    
    def propagate(self):

        self.update_position_and_velocity()
        self.update_electronic_part()
        self.update_nuclear_part()

        #print('TBF COEFFS', self.tbf_coeffs) ## Debug code
        #print('TBF COEFFS NORM', np.linalg.norm(self.tbf_coeffs)) ## Debug code
        print('TBF COEFFS ABS', np.abs(self.tbf_coeffs)**2) ## Debug code

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


    def make_tbf_coeffs_tderiv(self, t, tbf_coeffs):
        
        tbf_coeffs_tderiv = (-1.0j / H_DIRAC) * np.dot(
            np.linalg.inv(self.S_tbf), np.dot(self.H_tbf, tbf_coeffs.transpose())
        ).transpose()

        return tbf_coeffs_tderiv


    def update_nuclear_part(self):

        # construct TBF Hamiltonian

        n_tbf = self.get_total_tbf_count()

        self.S_tbf = np.zeros( (n_tbf, n_tbf), dtype = 'complex128' )
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

                S_ij = Tbf.get_wf_overlap(guy_i, guy_j, gaussian_overlap = g_ij)
                H_ij = Tbf.get_tbf_hamiltonian_element_BAT(guy_i, guy_j, gaussian_overlap = g_ij)
                #H_ij = 0.0 ## Debug code

                self.S_tbf[i_tbf,j_tbf] = S_ij
                self.H_tbf[i_tbf,j_tbf] = H_ij

                self.S_tbf[j_tbf,i_tbf] = S_ij
                self.H_tbf[j_tbf,i_tbf] = np.conj(H_ij)

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

                self.H_tbf[i_tbf,j_tbf] -= 1.0j * H_DIRAC * val
                self.H_tbf[j_tbf,i_tbf] += 1.0j * H_DIRAC * val

        # symmetrize S & hermitize H

        for i in range(n_tbf):
            for j in range(i, n_tbf):

                val = 0.5 * (self.S_tbf[i,j] + self.S_tbf[j,i])
                self.S_tbf[i,j] = val
                self.S_tbf[j,i] = val

                #val = 0.5 * ( self.H_tbf[i,j] + np.conjugate(self.H_tbf[j,i]) )
                #self.H_tbf[i,j] = val
                #self.H_tbf[j,i] = np.conjugate(val)

        # subtract energy origin

        if self.H_tbf_diag_origin is None:
            self.H_tbf_diag_origin = np.diag(self.H_tbf)[0]
        
        for i in range(n_tbf):
            self.H_tbf[i,i] -= self.H_tbf_diag_origin
        
        # propagation of TBF coeffs
        
        tbf_coeffs = self.get_tbf_coeffs()

        new_tbf_coeffs = self.integrator.engine(
            self.dt, 0.0, tbf_coeffs, self.make_tbf_coeffs_tderiv
        )

        #tbf_coeffs_tderiv = (-1.0j / H_DIRAC) * np.dot(
        #    np.linalg.inv(self.S_tbf), np.dot(self.H_tbf, tbf_coeffs.transpose())
        #).transpose()

        #print('H_TBF', self.H_tbf) ## Debug code
        #print('S_TBF', self.S_tbf) ## Debug code
        #print('TBF COEFFS TDERIV', tbf_coeffs_tderiv) ## Debug code

        self.set_new_tbf_coeffs(new_tbf_coeffs)
        
        ### Debug code
        #coh = self.tbfs[0].e_part.get_e_coeffs()[1].conj() * self.tbfs[1].e_part.get_e_coeffs()[2]
        #print('RHO 1,2', coh, abs(coh))
        ##print('E_COEFF 1', self.tbfs[0].e_part.get_e_coeffs()[1])
        ##print('E_COEFF 2', self.tbfs[1].e_part.get_e_coeffs()[2])
        ### End Debug code

        return
    

    def get_total_tbf_count(self):
        #return self.total_tbf_count
        return len(self.tbfs)


    def get_live_tbf_count(self):
        return self.live_tbf_count


    def get_tbf_coeffs(self):
        return deepcopy(self.tbf_coeffs)


    def get_tbf_coeffs_tderiv(self):
        return deepcopy(self.tbf_coeffs_tderiv)


    def get_old_tbf_coeffs(self):
        return deepcopy(self.old_tbf_coeffs)


    def set_new_tbf_coeffs(self, new_tbf_coeffs):
        
        self.old_tbf_coeffs = deepcopy(self.tbf_coeffs)
        self.tbf_coeffs     = new_tbf_coeffs

        return


    def set_new_tbf_coeffs_tderiv(self, new_tbf_coeffs_tderiv):
        
        self.old_tbf_coeffs_tderiv = deepcopy(self.tbf_coeffs_tderiv)
        self.tbf_coeffs_tderiv     = new_tbf_coeffs_tderiv

        return
