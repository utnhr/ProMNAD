#-*- coding: utf-8 -*-

import numpy as np
from tbfmodule import Tbf
from constants import ATOM_MASSES_AMU, DEFAULT_GWP_WIDTH_AU, H_DIRAC, AMU2AU, AU2SEC
from copy import deepcopy
from utils import stop_with_error
from settingsmodule import load_setting
from integratormodule import Integrator
from files import GlobalOutputFiles

class World:
    
    worlds = []


    def __init__(self, settings):
        
        self.settings = deepcopy(settings)

        self.live_tbf_count  = 0

        self.tbfs                  = []
        self.tbf_coeffs            = np.array([], dtype='complex128')
        self.tbf_coeffs_tderiv     = np.zeros_like(self.tbf_coeffs)
        self.old_tbf_coeffs        = None
        self.old_tbf_coeffs_tderiv = None

        self.dt = settings['dt']

        self.S_tbf = None
        self.H_tbf = None

        self.H_tbf_diag_origin = None

        self.world_id = len(World.worlds)

        self.integmethod = load_setting(settings, 'integrator')

        self.integrator = Integrator(self.integmethod)
        #self.integrator = Integrator('adams_moulton_2')
        #print('WARNING: TBF coeffs integrated with leapfrog method.') ## Debug code
        #self.integrator = Integrator('leapfrog') ## Debug code

        self.tbf_coeffs_are_trivial = load_setting(settings, 'tbf_coeffs_are_trivial')

        self.print_xyz_interval = load_setting(settings, 'print_xyz_interval')

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

        initial_e_coeffs = [ 0.0+0.0j for i in range(self.settings['n_estate']) ]

        for i_coeff in range(n_coeff):

            initial_e_coeffs[initial_estates[i_coeff]] = 1.0+0.0j
            ### Debug code
            #print('WARNING: INITIAL STATE NOT CORRECT')
            #initial_e_coeffs[1] = 1.0
            #initial_e_coeffs[2] = 1.0
            #initial_e_coeffs /= np.linalg.norm(initial_e_coeffs)
            ### End Debug code

        initial_e_coeffs = np.array(initial_e_coeffs) 

        initial_e_coeffs /= np.linalg.norm(initial_e_coeffs) # normalize

        initial_tbf = Tbf(
            self.settings, self.atomparams, position, n_dof, self.settings['n_estate'], len(self.tbfs),
            momentum = momentum, mass = mass, width = width, e_coeffs = initial_e_coeffs,
            is_fixed = is_fixed,
        )

        self.add_tbf(initial_tbf, coeff = 1.0+0.0j, normalize = False)

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
            self.H_tbf_diag_origin = deepcopy(np.diag(self.H_tbf)[0])
        
        for i in range(n_tbf):
            self.H_tbf[i,i] -= self.H_tbf_diag_origin
        
        # propagation of TBF coeffs
        
        tbf_coeffs = self.get_tbf_coeffs()

        if n_tbf == 1 and self.tbf_coeffs_are_trivial:

            new_tbf_coeffs = [ 1.0+0.0j ]
        
        else:

            new_tbf_coeffs = self.integrator.engine(
                self.dt, 0.0, tbf_coeffs, self.make_tbf_coeffs_tderiv
            )

        #print('H_TBF', self.H_tbf) ## Debug code
        #print('S_TBF', self.S_tbf) ## Debug code
        #print('TBF COEFFS TDERIV', tbf_coeffs_tderiv) ## Debug code

        self.set_new_tbf_coeffs(new_tbf_coeffs)
        
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


    def print_results(self):

        if self.print_xyz_interval != 0 and (self.istep % self.print_xyz_interval) == 0:

            self.print_tbf_coeff_and_popul()

        return


    def print_tbf_coeff_and_popul(self):

        tbf_coeff_file = self.globaloutput.tbf_coeff
        tbf_popul_file = self.globaloutput.tbf_popul

        t = self.dt * self.istep
        
        time_str = "STEP %12d T= %20.12f fs" % (self.istep, t*AU2SEC*1.0e15)
        
        tbf_coeff_file.write(time_str)
        tbf_popul_file.write(time_str)

        n_tbf = self.get_total_tbf_count()
        
        for i_tbf in range(n_tbf):

            tbf_coeff_file.write( " %20.12f+%20.12fj" % ( self.tbf_coeffs[i_tbf].real, self.tbf_coeffs[i_tbf].imag ) )
            tbf_popul_file.write( " %20.12f" % ( abs(self.tbf_coeffs[i_tbf])**2 ) )

        tbf_coeff_file.write("\n")
        tbf_popul_file.write("\n")
        
        return
