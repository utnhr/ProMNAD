 #!-*- coding: utf-8 -*-

import sys
import math
from cmath import exp, pi
import numpy as np
from copy import deepcopy
from constants import H_DIRAC, ANGST2AU, AU2EV, ONEOVER24, ONEOVER720
import utils

class Integrator:
    
    def __init__(self, method):

        self.n_hist = 5

        self.y_hist = [ None for i in range(self.n_hist) ] # new (i-1) -> old
        self.f_hist = [ None for i in range(self.n_hist) ] # new (i-1) -> old

        self.error_threshold = 1.0e-10
        self.mix_alpha = 1.0
        
        self.i_called = 0

        self.function_name = 'self.' + method

        self.engine = eval(self.function_name) # integrator engine (euler, leapfrog, etc.)

        return


    def reset(self):
        
        self.y_hist = [ None for i in range(self.n_hist) ] # new (i-1) -> old
        self.f_hist = [ None for i in range(self.n_hist) ] # new (i-1) -> old

        self.i_called = 0

        self.engine = eval(self.function_name) # integrator engine (euler, leapfrog, etc.)

        return


    def update_history(self, y, f):
        
        for i_hist in range(self.n_hist-1):
            
            self.y_hist[i_hist+1] = self.y_hist[i_hist]
            self.f_hist[i_hist+1] = self.f_hist[i_hist]

            self.y_hist[0] = y
            self.f_hist[0] = f

        self.i_called += 1

        return


    def euler(self, dt, t, y, f_func, *fargs):
        
        f = f_func(t, y, *fargs)

        y_new = y + dt * f
        
        self.update_history(y, f)
        
        return y_new
    

    def leapfrog(self, dt, t, y, f_func, *fargs):

        if self.i_called < 1:

            return self.euler(dt, t, y, f_func, *fargs)

        else:
            
            f = f_func(t, y, *fargs)
            
            y_new = self.y_hist[0] + 2.0 * dt * f # y(i-1) + 2*f(i) -> y(i+1)

        self.update_history(y, f)

        return y_new


    def adams_bashforth_2(self, dt, t, y, f_func, *fargs):

        if self.i_called < 1:

            return self.euler(dt, t, y, f_func, *fargs)

        else:

            f = f_func(t, y, *fargs)

            y_new = y + 0.5 * dt * (3.0*self.f_hist[0] - f)

        self.update_history(y, f)
        
        return y_new


    def adams_bashforth_4(self, dt, t, y, f_func, *fargs):

        if self.i_called < 3:

            return self.adams_bashforth_2(dt, t, y, f_func, *fargs)

        else:
            
            f = f_func(t, y, *fargs)

            y_new = y + ONEOVER24 * dt * (
                55.0*self.f_hist[0] - 59.0*self.f_hist[1] + 37.0*self.f_hist[2] - 9.0*self.f_hist[3]
            )

        self.update_history(y, f)
        
        return y_new


    def adams_moulton_2(self, dt, t, y, f_func, *fargs):

        if self.i_called < 1:

            return self.euler(dt, t, y, f_func, *fargs)

        else:

            f = f_func(t, y, *fargs)

            #print('F_HIST & F', self.f_hist[0], f) ## Debug code

            y1p = y + 0.5 * dt * (3.0*self.f_hist[0] - f) # predictor: 2-step Adams-Bashforth

            i_iter = 0
            
            while True:

                if i_iter > 0 and i_iter % 1000 == 0:
                    print('AM2 ITER', i_iter, f_func.__name__)

                f1p = f_func(t+dt, y1p, *fargs)

                y1c = y + 0.5 * dt * (f1p + f) # corrector

                error = np.linalg.norm(y1p - y1c)

                if error < self.error_threshold:

                    y_new = y1c

                    break

                else:

                    y1p = self.mix_alpha * y1c + (1.0 - self.mix_alpha) * y1p

                    i_iter += 1

            self.update_history(y, f)
        
        return y_new

    def adams_moulton_4(self, dt, t, y, f_func, *fargs):
        
        if self.i_called < 3:
            
            return self.adams_bashforth_2(dt, t, y, f_func, *fargs)

        else:
            
            f = f_func(t, y, *fargs)

            y1p = y + ONEOVER24 * dt * (
                55.0*self.f_hist[0] - 59.0*self.f_hist[1] + 37.0*self.f_hist[2] - 9.0*self.f_hist[3]
            ) # predictor: 4-step Adams-Bashforth

            i_iter = 0

            while True:

                if i_iter > 0 and i_iter % 1000 == 0:
                    print('AM4 ITER', i_iter, f_func.__name__)

                f1p = f_func(t+dt, y1p, *fargs)

                y1c = y + ONEOVER720 * dt * (
                    251.0*f1p + 646.0*f - 264.0*self.f_hist[0] + 106.0*self.f_hist[1] - 19.0*self.f_hist[2]
                ) # corrector

                error = np.linalg.norm(y1p - y1c)

                if error < self.error_threshold:

                    y_new = y1c

                    break

                else:

                    y1p = self.mix_alpha * y1c + (1.0 - self.mix_alpha) * y1p

                    i_iter += 1

            self.update_history(y, f)
        
        return y_new

