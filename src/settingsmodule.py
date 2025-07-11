#!-*- coding: utf-8 -*-

import utils

default = {
    'basis'     : 'configuration',
    'excitation': 'cis',
    'is_fixed'  : False,
    'dt_deriv'  : 0.01,
    'dtau': 0.1,
    'alpha': 1.0,
    'do_elec_interpol': False,
    'nbin_interpol_elec': 1,
    'do_tbf_interpol': False,
    'nbin_interpol_tbf': 1,
    'reconst_interval': -1,
    'print_xyz_interval': 0,
    'print_mo_tdnac_interval': -1,
    'print_mo_coeffs_interval': -1,
    'integrator': 'adams_moulton_2',
    'integmixer': 'diis',
    'mixererror': 1.0e-11,
    #'do_sc_propagation': False,
    'propagate_in_orthogonal_basis': False,
    'tbf_coeffs_are_trivial': False,
    'e_coeffs_are_trivial': False,
    'flush_interval':  2000,
    'cloning_rule': 'energy',
    'cloning_parameters': { 'popul_threshold': 0.1, 'e_threshold': 0.01, },
    'max_n_tbf': -1,

    'ao_basis': 'def2svp',
    'xc': 'b3lyp',
    'charge': 0,
    #'spin': 0,
    'check_stability': False,
    'level_shift': 0.0,
    'max_scf_cycle': 300,
    'reuse_scf_dm': True,
    'reuse_scf_mo': False,

    'calc_nonorthogonality_interval': -1,
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


def process_settings(settings):
    
    # 'do_interpol' keyword turns on/off both TBF and electronic interpolation

    if 'do_interpol' in settings.keys():

        settings['do_elec_interpol'] = settings['do_interpol']
        settings['do_tbf_interpol'] = settings['do_interpol']

    return
