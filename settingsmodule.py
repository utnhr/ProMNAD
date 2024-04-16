#!-*- coding: utf-8 -*-

import utils

default = {
    'basis'     : 'configuration',
    'excitation': 'cis',
    'is_fixed'  : False,
    'dt_deriv'  : 0.01,
    'reconst_interval': 100,
    'print_xyz_interval': 0,
    'integrator': 'adams_moulton_2',
}

    
def load_setting(settings, key):

    if type(key) is str:

        try:
            
            return settings[key]

        except KeyError:
            
            return default[key]

    elif hasattr(key, "__iter__"):

        if len(key) > 1:
        
            return load_setting(settings[key[0]], key[1:])

        elif len(key) == 1:

            return load_setting(settings, key[-1])

        else:

            utils.stop_with_error("Invalid key.\n")

    else:
        
        utils.stop_with_error("Invalid key.\n")

