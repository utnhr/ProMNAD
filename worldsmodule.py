#-*- coding: utf-8 -*-

import numpy as np
from tbfmodule import Tbf

class World:
    
    worlds = []


    @classmethod
    def init_new_world(cls, settings):

        new_world = World()
        cls.worlds.append(new_world)

        new_world.settings = settings

        return


    def __init__(self):
        
        self.settings = None

        self.total_tbf_count = 0
        self.live_tbf_count  = 0

        tbfs                  = []
        tbf_coeffs            = np.array([], dtype='complex128')
        tbf_coeffs_tderiv     = np.zeros_like(self.tbf_coeffs)
        old_tbf_coeffs        = np.zeros_like(self.tbf_coeffs)
        old_tbf_coeffs_tderiv = np.zeros_like(self.tbf_coeffs)

        dt = 0.0

        S_tbf = None
        H_tbf = None
        
        return

    
    def propagate(self):

        self.update_position_and_velocity()
        self.update_electronic_part()
        self.update_nuclear_part()

        return
    

    def add_tbf(self, tbf, coeff=None):

        self.tbfs.append(tbf)

        if coeff is None:
    
            np.append(self.tbf_coeffs, 0.0+0.0j)
            np.append(self.tbf_coeffs_tderiv, 0.0+0.0j)

        else:

            np.append(self.tbf_coeffs, coeff)
            np.append(self.tbf_coeffs_tderiv, 0.0+0.0j)

            self.tbf_coeffs /= np.linalg.norm(self.tbf_coeffs)

        self.total_tbf_count += 1
        self.live_tbf_count  += 1

        return

    
    def destroy_tbf(self, tbf):
        
        self.live_tbf_count -= 1

        return


    def update_position_and_velocity(self):
        
        for guy in self.tbfs():

            if not guy.is_alive:
                continue
            
            guy.update_position_and_velocity(self.dt)

        return
    

    def update_electronic_part(self):

        for guy in self.tbfs(): # 'guy' is an individual tbf
            
            if not guy.is_alive:
                continue

            guy.update_electronic_part(self.dt)

        return


    def update_nuclear_part(self):

        # update TBF coeffs (leapfrog)

        old_tbf_coeffs      = self.get_old_tbf_coeffs()
        tbf_coeffs_tderiv   = self.get_tbf_coeffs_tderiv()

        tbf_coeffs = old_tbf_coeffs + 2.0 * tbf_coeffs_tderiv * dt

        self.set_new_tbf_coeffs(tbf_coeffs)
        
        # construct TBF Hamiltonian

        n_tbf = self.get_total_tbf_count()

        self.S_tbf = np.zeros( (n_tbf, n_tbf) )
        self.H_tbf = np.zeros( (n_tbf, n_tbf) )

        for i_tbf in range(n_tbf):
            
            guy_i = self.tbfs[i_tbf]

            if not guy_i.is_alive:
                continue

            for j_tbf in range(n_tbf):

                guy_j = self.tbfs[j_tbf]

                if not guy_j.is_alive:
                    continue

                g_ij = Tbf.get_gaussian_overlap(guy_i, guy_j)

                S_ij = Tbf.get_wf_overlap(guy_i, guy_j, gaussian_overlap = g_ij)
                H_ij = Tbf.get_tbf_hamiltonian_element_BAT(guy_i, guy_j, gaussian_overlap = g_ij)

                self.S_tbf[i,j] = S_ij
                self.H_tbf[i,j] = H_ij

        # < \psi_m | d/dt | \psi_n >

        for i_tbf in range(n_tbf):
            
            guy_i = self.tbfs[i_tbf]

            if not guy_i.is_alive:
                continue

            for j_tbf in range(n_tbf):

                guy_j = self.tbfs[j_tbf]

                if not guy_j.is_alive:
                    continue

                g_ij = Tbf.get_gaussian_overlap(guy_i, guy_j)

                val = Tbf.get_tbf_derivative_coupling(guy_i, guy_j, g_ij)

                H[i,j] -= val

        # symmetrize S & hermitize H

        for i in range(n_tbf):
            for j in range(i, n_tbf):

                val = 0.5 * (S[i,j] + S[j,i])
                S[i,j] = val
                S[j,i] = val

                val = 0.5 * ( H[i,j] + np.conjugate(H[j,i]) )
                H[i,j] = val
                H[j,i] = np.conjugate(val)
        
        # time derivative of TBF coeffs
        
        tbf_coeffs = self.get_tbf_coeffs()

        tbf_coeffs_tderiv = (-1.0j / H_DIRAC) * np.dot(
            np.linalg.inv(S), np.dot(H, tbf_coeffs)
        )

        self.set_new_tbf_coeffs_tderiv(tbf_coeffs_tderiv)

        return
    

    def get_total_tbf_count(self):
        return self.total_tbf_count


    def get_live_tbf_count(self):
        return self.live_tbf_count


    def get_tbf_coeffs(self):
        return tbf_coeffs


    def get_tbf_coeffs_tderiv(self):
        return tbf_coeffs_tderiv


    def set_new_tbf_coeffs(self, new_tbf_coeffs):
        
        self.old_tbf_coeffs = self.tbf_coeffs
        self.tbf_coeffs     = new_tbf_coeffs

        return


    def set_new_tbf_coeffs_tderiv(self, new_tbf_coeffs_tderiv):
        
        self.old_tbf_coeffs_tderiv = self.tbf_coeffs_tderiv
        self.tbf_coeffs_tderiv     = self.tbf_coeffs_tderiv

        return
