#!-*- coding: utf-8 -*-

import sys
import os
import inspect
import numpy as np
import time

import constants


class Timer:
    
    checkpoint_times = {}
    
    @classmethod
    def set_checkpoint_time(cls, name, return_laptime = False):
        
        current_time = time.perf_counter()

        try:

            previous_time = cls.checkpoint_times[name]

        except KeyError:
            
            previous_time = current_time

        cls.checkpoint_times[name] = current_time

        if return_laptime:
            
            return current_time - previous_time

        else:

            return current_time


    @classmethod
    def get_checkpoint_time(cls, name, create_if_absent = False):

        try:

            checkpoint_time = cls.checkpoint_times[name]

        except KeyError:

            cls.set_checkpoint_time(name)

            checkpoint_time = cls.checkpoint_times[name]
        
        return checkpoint_time


    @classmethod
    def get_time(cls):
        
        return time.perf_counter()


class Printer:

    outfile = sys.stdout
    errfile = sys.stderr


    @classmethod
    def set_outfile(cls, new_outfile):
        
        cls.outfile = new_outfile


    @classmethod
    def set_errfile(cls, new_errfile):
        
        cls.errfile = new_errfile


    @classmethod
    def write_out(cls, msg, flush=False):
    
        cls.outfile.write(msg)

        if flush:
            flush(cls.outfile)


    @classmethod
    def write_err(cls, msg, flush=False):
        
        cls.errfile.write(msg)

        if flush:
            flush(cls.errfile)


def stop_with_error(msg):

    frame = inspect.currentframe().f_back

    filename = os.path.basename(frame.f_code.co_filename)
    funcname = frame.f_code.co_name
    lineno   = frame.f_lineno
    
    Printer.write_err(
        " ERROR: %s\n File    : %s\n Line no.: %d\n Function: %s\n" % (
            msg, filename, lineno, funcname
        )
    )
    sys.exit(1)


def warn(msg):

    frame = inspect.currentframe().f_back

    filename = os.path.basename(frame.f_code.co_filename)
    funcname = frame.f_code.co_name
    lineno   = frame.f_lineno
    
    Printer.write_err(
        " WARNING: %s\n File    : %s\n Line no.: %d\n Function: %s\n" % (
            msg, filename, lineno, funcname
        )
    )

    return


def is_equal_scalar(a, b, eps = constants.EPS_FLOAT_EQUAL):
    
    return abs(a-b) < eps


def is_equal_ndarray(a, b, eps = constants.EPS_FLOAT_EQUAL):
    
    return np.max(np.abs(a-b)) < eps


def coord_1d_to_2d(coord_1d):
    
    natom = int(coord_1d.size / 3)

    return coord_1d.reshape([natom, 3])


def coord_1d_to_pyscf(coord_1d, elems):
    
    natom = int(coord_1d.size / 3)

    coords = coord_1d.reshape([natom, 3])

    atoms = []

    for elem, coord in zip(elems, coords):

        atoms.append( [ elem, coord ] )

    return atoms


def symmetrize(M, is_upper_triangle = True): ## placeholder

    if is_upper_triangle:
        temp = np.triu(M)
    else:
        temp = np.tril(M)

    res = temp + temp.transpose() - np.diag(np.diag(temp))
    
    return res


def hermitize(M, is_upper_triangle = True): ## placeholder

    if is_upper_triangle:
        temp = np.triu(M)
    else:
        temp = np.tril(M)
    
    res = temp + np.conjugate( temp.transpose() ) - np.diag(np.diag(temp))

    return res


def check_time_equal(t1, t2, action = 'stop'):
    
    if not is_equal_scalar(t1, t2):

        frame = inspect.currentframe().f_back

        filename = os.path.basename(frame.f_code.co_filename)
        funcname = frame.f_code.co_name
        lineno   = frame.f_lineno

        diff = t1 - t2
        msg = "Two or more physical quantities, which must be at the same time points, are at different time points (t1-t2= %20.12f)." % diff
        
        if action == 'stop':

            Printer.write_err(
                " ERROR: %s\n File    : %s\n Line no.: %d\n Function: %s\n" % (
                    msg, filename, lineno, funcname
                )
            )
            sys.exit(1)

        elif action == 'warn':

            Printer.write_err(
                " WARNING: %s\n File    : %s\n Line no.: %d\n Function: %s\n" % (
                    msg, filename, lineno, funcname
                )
            )

        else:

            stop_with_error("Unknown action %s specified." % action)

    else:

        return


def binary_search(values, target):
    
    n_values = len(values)

    i_min = 0
    i_max = n_values-1

    while True:

        if i_max - i_min < 2:
            break

        i_half = (i_min + i_max) // 2

        if   values[i_half] < target:

            i_min = i_half

        elif values[i_half] > target:
            
            i_max = i_half

        elif values[i_half] == target:
            
            i_min = i_half
            i_max = i_half

    return i_min, i_max
