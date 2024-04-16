#!-*- coding: utf-8 -*-

from utils import stop_with_error
from interface_dftbplus import dftbplus_manager
from inout import get_geom, get_traject
import os

def configure_calculation(settings):
    
    if settings['calctype'] == 'config-aimc':

        if settings['mo_type'] == 'restricted':
            
            settings['n_occ'] = len(settings['active_occ_mos'])

            settings['n_estate'] = settings['n_occ'] * len(settings['active_vir_mos']) + 1

    else:

        stop_with_error("Unknown calctype %s ." % settings['calctype'])

    if settings['read_traject']:

        settings['given_geoms'], settings['given_velocities'] = get_traject(
            'traject', 'velocity', return_geom_1d = True
        )

def fill_default_settings(settings):
    
    keys = settings.keys()

    if 'title' not in keys:

        settings['title'] = 'No title'

    return

def init_qc_engine(settings, workdir_suffix, **kwords):
    
    if settings['engine']['type'] == 'dftb+':

        elems, position = get_geom(settings['geom_file'], return_geom_1d = True, return_elems_1d = True)

        workdir = os.path.join(settings['engine']['workdir'], workdir_suffix)
        
        try:
            os.mkdir(workdir)
        except:
            pass

        home = os.getcwd()

        os.chdir(workdir)
        dftbplus_instance = dftbplus_manager(
            settings['engine']['libpath'], workdir, elems, position,
        )
        os.chdir(home)

        return dftbplus_instance

    else:

        stop_with_ettor("Unknown quantum chemistry package %s ." % settings['engine']['type'])
