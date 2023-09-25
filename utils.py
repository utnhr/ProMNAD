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
    def set_outfile(new_outfile):
        
        outfile = new_outfile


    @classmethod
    def set_errfile(new_errfile):
        
        errfile = new_errfile


    @classmethod
    def write_out(msg, flush=False):
    
        outfile.write(msg)

        if flush:
            flush(outfile)


    @classmethod
    def write_err(msg, flush=False):
        
        errfile.write(msg)

        if flush:
            flush(errfile)


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
