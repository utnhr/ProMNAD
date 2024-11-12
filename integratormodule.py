 #!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
from copy import deepcopy
from constants import H_DIRAC, ANGST2AU, AU2EV, ONEOVER24, ONEOVER720
import utils

class Integrator:
    
    def __init__(self, method, error_threshold = 1.0e-10, mix_alpha = 0.5, mode = 'passing'):

        self.max_hist = 5

        self.y_hist = [ None for i in range(self.max_hist) ] # new (i-1) -> old
        self.f_hist = [ None for i in range(self.max_hist) ] # new (i-1) -> old

        self.error_threshold = error_threshold
        self.mix_alpha = mix_alpha
        
        #self.i_called = 0
        self.n_hist = 0

        self.function_name = 'self.' + method

        #self.engine = eval(self.function_name) # integrator engine (euler, leapfrog, etc.)

        if mode == 'passing': # return y(t+Dt) from y(t) and fargs(t)

            self.is_chasing_mode = False

        elif mode == 'chasing': # return y(t+Dt) from fargs(t+Dt); y(0) and f(0) must be saved at the initial step

            self.is_chasing_mode = True

            self.is_history_initialized = False

        else:

            utils.stop_with_error("Unknown integrator mode %s ." % mode)

        return


    def engine(self, dt, t, y, f_func, *fargs):
        # y @ t
        # f_func & *fargs @ t or t+Dt (for passing and chasing mode, respectively)

        if self.is_chasing_mode and not self.is_history_initialized:

            utils.stop_with_error('initialize_history must be called in advance for chasing mode.')

        if not self.is_chasing_mode:

            f = f_func(t, y, *fargs) # y @ t, fargs @ t -> f @ t

            self.update_history(y, f)
        
        # history of at this point
        #
        #   y_hist[0] @ t, y_hist[1] @ t-Dt, ...
        #   f_hist[0] @ t, f_hist[1] @ t-Dt, ...

        y_new = eval(self.function_name)(dt, t, f_func, *fargs)

        if self.is_chasing_mode:

            f_new = f_func(t+dt, y_new, *fargs) # y_new @ t+Dt, fargs @ t+Dt -> f_new @ t+Dt

            self.update_history(y_new, f_new)
        
        #self.i_called += 1

        return y_new


    def reset(self):
        
        self.y_hist = [ None for i in range(self.max_hist) ] # new (i-1) -> old
        self.f_hist = [ None for i in range(self.max_hist) ] # new (i-1) -> old

        self.n_hist = 0
        
        if self.is_chasing_mode:
            self.is_history_initialized = False

        #self.engine = eval(self.function_name) # integrator engine (euler, leapfrog, etc.)

        return


    def initialize_history(self, t, y, f_func, *fargs):

        f = f_func(t, y, *fargs)

        self.update_history(y, f)

        self.is_history_initialized = True

        return


    def update_history(self, y, f):
        
        for i_hist in range(self.max_hist-1):
            
            self.y_hist[i_hist+1] = self.y_hist[i_hist]
            self.f_hist[i_hist+1] = self.f_hist[i_hist]

        self.y_hist[0] = deepcopy(y)
        self.f_hist[0] = deepcopy(f)

        self.n_hist = min(self.n_hist+1, self.max_hist)

        return

    
    def euler(self, dt, t, f_func, *fargs):
        
        y_new = self.y_hist[0] + dt * self.f_hist[0]
        
        return y_new


    def leapfrog(self, dt, t, f_func, *fargs):

        if self.n_hist < 2:

            return self.euler(dt, t, f_func, *fargs)

        else:
            
            y_new = self.y_hist[1] + 2.0 * dt * self.f_hist[0] # y(i-1) + 2*f(i) -> y(i+1)

        return y_new


    def adams_bashforth_2(self, dt, t, f_func, *fargs):

        if self.n_hist < 2:

            return self.euler(dt, t, f_func, *fargs)

        else:

            y_new = self.y_hist[0] + 0.5 * dt * (3.0*self.f_hist[1] - self.f_hist[0])

        return y_new


    def adams_bashforth_4(self, dt, t, f_func, *fargs):

        if self.n_hist < 5:

            return self.adams_bashforth_2(dt, t, f_func, *fargs)

        else:
            
            y_new = self.y_hist[0] + ONEOVER24 * dt * (
                55.0*self.f_hist[1]- 59.0*self.f_hist[2] + 37.0*self.f_hist[3] - 9.0*self.f_hist[4]
            )

        return y_new


    def adams_moulton_2(self, dt, t, f_func, *fargs):

        if not self.is_chasing_mode:
            utils.stop_with_error('Implicit solvers only available for chasing mode.')

        if self.n_hist < 2:

            y1p = self.y_hist[0] + dt * self.f_hist[0] # predictor: Euler

        else:

            y1p = self.y_hist[0] + 0.5 * dt * (3.0*self.f_hist[1] - self.f_hist[0]) # predictor: 2-step Adams-Bashforth

        i_iter = 0
        
        while True:

            if i_iter > 0 and i_iter % 1000 == 0:
                print('AM2 ITER', i_iter, f_func.__name__)

            f1p = f_func(t+dt, y1p, *fargs)

            y1c = self.y_hist[0] + 0.5 * dt * (f1p + self.f_hist[0]) # corrector

            error = np.linalg.norm(y1p - y1c)

            #print('AM2 ITER', i_iter, error, f_func.__name__)

            if error < self.error_threshold:

                y_new = y1c

                break

            else:

                y1p = self.mix_alpha * y1c + (1.0 - self.mix_alpha) * y1p

                i_iter += 1

        #print('AM2 ITER', i_iter, f_func.__name__) ## Debug code
        return y_new


    def adams_moulton_4(self, dt, t, f_func, *fargs):

        if not self.is_chasing_mode:
            utils.stop_with_error('Implicit solvers only available for chasing mode.')

        if self.n_hist < 2:
            y1p = self.y_hist[0] + dt * self.f_hist[0] # predictor: Euler

        elif self.n_hist< 4:
            
            y1p = self.y_hist[0] + 0.5 * dt * (3.0*self.f_hist[1] - self.f_hist[0]) # predictor: 2-step Adams-Bashforth

        else:

            y1p = self.y_hist[0] + ONEOVER24 * dt * (
                55.0*self.f_hist[0] - 59.0*self.f_hist[1] + 37.0*self.f_hist[2] - 9.0*self.f_hist[3]
            ) # predictor: 4-step Adams-Bashforth
            
        i_iter = 0

        while True:

            if i_iter > 0 and i_iter % 1000 == 0:
                print('AM4 ITER', i_iter, f_func.__name__)

            f1p = f_func(t+dt, y1p, *fargs)

            y1c = self.y_hist[0] + ONEOVER720 * dt * (
                251.0*f1p + 646.0*self.f_hist[0] - 264.0*self.f_hist[1] + 106.0*self.f_hist[2] - 19.0*self.f_hist[3]
            ) # corrector

            error = np.linalg.norm(y1p - y1c)

            if error < self.error_threshold:

                y_new = y1c

                break

            else:

                y1p = self.mix_alpha * y1c + (1.0 - self.mix_alpha) * y1p

                i_iter += 1

        return y_new
