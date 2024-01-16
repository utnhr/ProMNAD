#!-*- coding: utf-8 -*-

from utils import stop_with_error
from interface_dftbplus import dftbplus_manager
from inout import get_geom

def configure_calculation(settings):
    
    if settings['calctype'] == 'config-aimc':

        if settings['mo_type'] == 'restricted':
            
            settings['n_occ'] = len(settings['active_occ_mos'])

            settings['n_estate'] = settings['n_occ'] * len(settings['active_vir_mos']) + 1

    else:

        stop_with_error("Unknown calctype %s ." % settings['calctype'])

def fill_default_settings(settings):
    
    keys = settings.keys()

    if 'title' not in keys:

        settings['title'] = 'No title'

    return

def init_qc_engine(settings):
    
    if settings['engine']['type'] == 'dftb+':

        elems, position = get_geom(settings['geom_file'], return_geom_1d = True, return_elems_1d = True)

        dftbplus_manager.dftbplus_init(
            settings['engine']['libpath'], settings['engine']['workdir'], elems, position,
        )

    else:

        stop_with_ettor("Unknown quantum chemistry package %s ." % settings['engine']['type'])
