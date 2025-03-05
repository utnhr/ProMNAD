#!-*- coding: utf-8 -*-

def propagate(propagator_type, dt, r1=None, v1=None, r0=None, v0=None):
    
    if propagator_type == 'leapfrog':

        return r0 + 2.0*v1*dt

    else:
        
        msg = "Unknown propagator type %s .\n" % propagator_type
        
        utils.stop_with_error(msg)

    return
