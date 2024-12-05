#!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
from copy import deepcopy
from constants import H_DIRAC, ANGST2AU, AU2EV, ONEOVER24, ONEOVER720
import utils

class Mixer:
    

    def __init__(self, method, max_hist=100, almix=0.3):

        self.max_hist = max_hist

        self.function_name = 'self.' + method

        self.almix = almix # mixing parameter
        
        self.reset()
        
        return


    def update_history(self, y=None, error=None):
        
        for i_hist in range(self.max_hist-2,-1,-1):
            
            if y is not None:
                self.y_hist[i_hist+1] = deepcopy(self.y_hist[i_hist])

            if error is not None:
                self.error_hist[i_hist+1] = deepcopy(self.error_hist[i_hist])

        self.y_hist[0] = deepcopy(y)
        self.error_hist[0] = deepcopy(error)

        self.n_hist = min(self.n_hist+1, self.max_hist)

        return


    def engine(self, y, error=None):

        self.update_history(y, error)
        
        y_new = eval(self.function_name)()

        return y_new


    def reset(self):

        self.n_hist = 0

        self.y_hist     = [ None for i in range(self.max_hist) ] # new (i-1) -> old
        self.error_hist = [ None for i in range(self.max_hist) ] # new (i-1) -> old

        return


    def simple(self):

        if self.n_hist < 2:

            y_new = self.y_hist[0]

        else:

            y_new = self.almix * self.y_hist[0] + (1.0 - self.almix) * self.y_hist[1]

        return y_new


    def diis(self):
        
        B = np.zeros( (self.n_hist+1, self.n_hist+1), dtype = 'float64' )

        rhs = np.zeros( self.n_hist+1, dtype = 'float64')
        rhs[self.n_hist] = 1.0

        #print('ERROR HIST', self.error_hist) ## Debug code

        for i in range(self.n_hist):
            for j in range(i+1):

                B[i,j] = np.dot( self.error_hist[i].flatten(), self.error_hist[j].flatten() )
                B[j,i] = B[i,j]

        for i in range(self.n_hist):
            B[self.n_hist, i] = 1.0
            B[i, self.n_hist] = 1.0

        B[self.n_hist, self.n_hist] = 0.0

        #print('B MATRIX', B) ## Debug code

        c_vec = np.dot( np.linalg.inv(B), rhs )

        y_new = 0.0

        for i in range(self.n_hist):

            y_new += c_vec[i] * self.y_hist[i]
            #print(c_vec[i]) ## Debug code

        return y_new
