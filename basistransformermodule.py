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
    def ao2mo(cls, M, C, is_C_columnmajor = False):
        
        # M: target matrix in AO
        # C: set of orthonormal orbitals
        # For example ...
        # if is_C_columnmajor is True:
        #   each column contains a set of MO coefficients for a MO
        # else:
        #   each row contains a set of ...

        if is_C_columnmajor:
            basis = C
        else:
            basis = C.transpose()

        return np.dot( basis.transpose().conjugate(), np.dot(M, basis) )


    @classmethod
    def mo2ao(cls, M, S, C, is_C_columnmajor = False):
        
        # M: target matrix in MO (C)
        # S: AO overlap matrix
        # C: set of orthonormal orbitals
        # For example ...
        # if is_C_columnmajor is True:
        #   each column contains a set of MO coefficients for a MO
        # else:
        #   each row contains a set of ...

        if is_C_columnmajor:
            SC = np.dot( S, C.transpose() )
        else:
            SC = np.dot(S, C)

        return np.dot( SC.transpose().conjugate(), np.dot(M, SC) )


    @classmethod
    def moinmo(cls, MOinAO, S, C, is_C_columnmajor = False):
        
        #if is_C_columnmajor:
        #    SC = np.dot( S, C.transpose() )
        #else:
        #    SC = np.dot( S, C )
        if is_C_columnmajor:
            CdagS = np.dot( C.transpose(), S )
        else:
            CdagS = np.dot( C, S )

        return np.dot( CdagS, MOinAO.transpose() ).transpose()


    @classmethod
    def moinao(cls, MOinC, S, C, is_C_columnmajor = False):
        
        #if is_C_columnmajor:
        #    SC = np.dot( S, C.transpose() )
        #else:
        if is_C_columnmajor:
            CdagS = np.dot( C.transpose(), S )
        else:
            CdagS = np.dot( C, S )

        return np.dot( np.linalg.inv(CdagS), MOinC.transpose() ).transpose()
    

    @classmethod
    def lowdin(cls, M):
        
        #U, S, Vh = np.linalg.svd(M, compute_uv = True)

        #L = np.dot(U, Vh) # L[AO,AO]

        ## make L row major to be consistent with MO coefficients
        #
        #if return_singular_values:
        #    return L.transpose(), S
        #else:
        #    return L.transpose()
        
        # lambda = VSV
        lmd, V = np.linalg.eig(M)
        
        # K = diag(1/sqrt(lambda))
        K = np.diag( 1.0 / np.sqrt(lmd) )

        # S^(-1/2) = VKV^\dag
        Sisq = np.dot( V, np.dot( K, V.transpose().conjugate() ) )
            
        return Sisq.transpose()
