#-*- coding: utf-8 -*-

import numpy as np

class World:
    

    def __init__(self):
        
        self.settings = None

        self.total_tbf_count = 0
        self.live_tbf_count  = 0

        tbfs                  = []
        tbf_coeffs            = np.array([], dtype='complex128')
        tbf_coeffs_tderiv     = np.zeros_like(self.tbf_coeffs)
        old_tbf_coeffs        = np.zeros_like(self.tbf_coeffs)
        old_tbf_coeffs_tderiv = np.zeros_like(self.tbf_coeffs)
        
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
