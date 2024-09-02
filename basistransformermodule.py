 #!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
from copy import deepcopy
from constants import H_DIRAC, ANGST2AU, AU2EV, ONEOVER24, ONEOVER720
import utils

class BasisTransformer:


    @classmethod
    def ao2mo(cls, M, C, is_C_columnmajor = True):
        
        # M: target matrix in AO
        # C: set of orthonormal orbitals
        # For example ...
        # if is_C_columnmajor is True:
        #   each column contains a set of MO coefficients for a MO
        # else:
        #   each row contains a set of ...

        if is_C_columnmajor:
            basis = C.transpose()
        else:
            basis = C

        return np.dot( basis.transpose().conjugate(), np.dot(M, basis) )


    @classmethod
    def mo2ao(cls, M, S, C, is_C_columnmajor = True):
        
        # M: target matrix in MO (C)
        # S: AO overlap matrix
        # C: set of orthonormal orbitals
        # For example ...
        # if is_C_columnmajor is True:
        #   each column contains a set of MO coefficients for a MO
        # else:
        #   each row contains a set of ...

        if is_C_columnmajor:
            SC = np.dot(S, C)
        else:
            SC = np.dot( S, C.transpose() )

        return np.dot( SC.transpose().conjugate(), np.dot(M, SC) )
    

    @classmethod
    def lowdin(cls, M, return_singular_values = False):
        
        U, S, Vh = np.linalg.svd(M, compute_uv = True)

        L = np.dot(U, Vh) # L[AO,AO]
        
        if return_singular_values:
            return L, S
        else:
            return L
