#!-*- coding: utf-8 -*-

import utils

default = {
    'basis'     : 'configuration',
    'excitation': 'cis',
    'is_fixed'  : False,
    'dt_deriv'  : 0.01,
    'dt': 0.1,
    'do_interpol': True,
    'nbin_interpol': 10,
    'reconst_interval': -1,
    'print_xyz_interval': 0,
    'integrator': 'adams_moulton_2',
    'tbf_coeffs_are_trivial': False,
    'flush_interval':  2000,
    'cloning_rule': 'energy',
    'cloning_parameters': { 'popul_threshold': 0.1, 'e_threshold': 0.01, },
    'max_n_tbf': -1,
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

