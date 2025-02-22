 #!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
from copy import deepcopy
from constants import H_DIRAC, ANGST2AU, AU2EV, ONEOVER24, ONEOVER720
import utils
from mixermodule import Mixer

class UnitaryPropagator:

    
    methods = [ 'sc_unitary', 'unitary', ]

    
    def __init__(self, method, mixertype='diis', error_threshold=1.0e-11):

        self.function_name = 'self.' + method

        self.reset()

        self.mixer = Mixer(mixertype) # to be made optional
        #self.mixer = Mixer('simple') # to be made optional

        self.error_threshold = error_threshold
        
        return


    def engine(self, dt, y, f_func, *fargs):
        
        U = self.get_propagator(dt, y, f_func, *fargs)

        #return np.dot(U, y)
        # MO coeffs are column major
        return np.dot( U, y.transpose() ).transpose()


    def update_history(self, M):
        
        self.history[1] = deepcopy(self.history[0])
        self.history[0] = deepcopy(M)

        self.n_hist = 1

        return


    def reset(self):
        
        self.history = [ None, None ] # 0: new, 1: old
        self.n_hist = 0

        return


    def get_propagator(self, dt, y, f_func, *fargs):
        
        U = eval(self.function_name)(dt, y, f_func, *fargs)

        return U


    def get_matrix(self, dt, y, f_func, *fargs):

        # propagator: exp( -i M dt / \hbar )
        # typically, M is S^-1 * H
        
        M = f_func(y, *fargs)

        return M


    def matrix_exp(self, M, trancation_order = 100):

        n = np.shape(M)[0]
        mat_type = M.dtype

        I = np.identity(n, dtype = mat_type)

        res = deepcopy(I)
        
        for i_order in range(trancation_order, 0, -1):

            res = I + ( 1 / float(i_order) ) * np.dot(M, res)

        return res


    def get_self_consistent_propagator(self, method_propagator, dt, y, f_func, *fargs):

        # \int^t+Dt_t exp(-iHt) dt ~ exp(-iH(t+Dt)Dt/2) * exp(-iH(t)Dt/2)

        yp = deepcopy(y)

        if self.n_hist < 1:

            new_propagator = method_propagator(dt/2, yp, f_func, *fargs)
            
            old_propagator = deepcopy(new_propagator)

        else:

            #old_propagator = self.history[1]
            old_propagator = deepcopy(self.history[0])

            yp = deepcopy(y)

            i_iter = 0
            
            while True:

                new_propagator = method_propagator(dt/2, yp, f_func, *fargs)

                new_yp = np.dot( np.dot(new_propagator, old_propagator), y.transpose() ).transpose()

                #error_vec = (new_yp - yp).real
                #error_vec = np.abs(new_yp - yp)
                error_vec = np.abs(new_yp)**2 - np.abs(yp)**2
                #error_vec = np.abs(new_yp) - np.abs(yp)

                error = np.max( np.abs(error_vec) )
                #error = np.sqrt( np.max( np.abs(error_vec) ) )

                print('PROPAGATOR ITER', i_iter, error, f_func.__name__)
                #print(old_propagator) ## Debug code
                #print(new_propagator) ## Debug code
                #print(yp[0][0]) ## Debug code
                #print(new_yp[0][0]) ## Debug code

                if error < self.error_threshold:

                    yp = deepcopy(new_yp)

                    self.mixer.reset()

                    break

                else:

                    yp = self.mixer.engine(new_yp, error_vec)

                    i_iter += 1

        self.update_history(new_propagator)
        
        res = np.dot(new_propagator, old_propagator)

        return  res


    def sc_unitary(self, dt, y, f_func, *fargs):
        
        return self.get_self_consistent_propagator(self.unitary, dt, y, f_func, *fargs)
    

    def unitary(self, dt, y, f_func, *fargs):

        #M = -1.0j * dt * np.dot(Sinv, H)
        M = (0.0-1.0j) * dt * f_func(y, *fargs)

        return self.matrix_exp(M)


    #def crank_nicholson(self, Sinv, H, dtau):
    #    
    #    M = -1.0j * np.dot(Sinv, H) * dtau * 0.5

    #    n = np.shape(M)[0]
    #    mat_type = M.dtype

    #    I = np.identity(n, dtype = mat_type)

    #    A = I - 1.0j * M - float(1.0/2.0) * np.dot(M, M) + 1.0j * float(1.0/6.0) * np.dot(M, np.dot(M, M))
    #    B = I + 1.0j * M - float(1.0/2.0) * np.dot(M, M) - 1.0j * float(1.0/6.0) * np.dot(M, np.dot(M, M))
    #    Binv = np.linalg.inv(B)

    #    return np.dot(Binv, A)
