#!-*- coding: utf-8 -*-

import sys
import os
import inspect
import numpy as np

import constants


class printer:

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
    
    printer.write_err(
        " ERROR: %s\n File    : %s\n Line no.: %d\n Function: %s\n" % (
            msg, filename, lineno, funcname
        )
    )
    sys.exit(1)


def is_equal_scalar(a, b):
    
    return abs(a-b) < constants.EPS_FLOAT_EQUAL


def is_equal_ndarray(a, b):
    
    return np.max(np.abs(a-b)) < constants.EPS_FLOAT_EQUAL


def coord_1d_to_2d(coord_1d):
    
    natom = int(coord_1d.size / 3)

    return coord_1d.reshape([natom, 3])

def symmetrize(M, is_upper_triangle = True): ## placeholder

    dim = M.shape[0]
    
    if is_upper_triangle:
        temp = np.triu(M)
    else:
        temp = np.tril(M)

    temp = temp + temp.transpose()
    
    for i in range(dim):
        temp[i,i] *= 0.5

    return temp

def hermitize(M, is_upper_triangle = True): ## placeholder

    dim = M.shape[0]
    
    if is_upper_triangle:
        temp = np.triu(M)
    else:
        temp = np.tril(M)

    temp = temp + np.conjugate( temp.transpose() )
    
    for i in range(dim):
        temp[i,i] *= 0.5

    return temp
