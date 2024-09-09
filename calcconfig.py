#!-*- coding: utf-8 -*-

from utils import stop_with_error
from settingsmodule import load_setting
from inout import get_geom, get_traject
import os

def configure_calculation(settings):
    
    if load_setting(settings, 'calctype') == 'config-aimc':

        if load_setting(settings, 'mo_type') == 'restricted':
            
            settings['n_occ'] = len( load_setting(settings, 'active_occ_mos') )

            settings['n_estate'] = load_setting(settings, 'n_occ') * len( load_setting(settings, 'active_vir_mos') ) + 1

    else:

        stop_with_error("Unknown calctype %s ." % load_setting(settings, 'calctype'))

    if load_setting(settings, 'read_traject'):

        settings['given_geoms'], settings['given_velocities'] = get_traject(
            'traject', 'velocity', return_geom_1d = True
        )

def fill_default_settings(settings):
    
    keys = settings.keys()

    if 'title' not in keys:

        settings['title'] = 'No title'

    return

def init_qc_engine(settings, workdir_suffix, **kwords):
    
    if load_setting( settings, ('engine', 'type') ) == 'dftb+':

        from interface_dftbplus import dftbplus_manager

        elems, position = get_geom(load_setting(settings, 'geom_file'), return_geom_1d = True, return_elems_1d = True)

        workdir = os.path.join( load_setting( settings, ('engine', 'workdir') ), workdir_suffix )
        
        try:
            os.mkdir(workdir)
        except:
            pass

        home = os.getcwd()

        os.chdir(workdir)
        dftbplus_instance = dftbplus_manager(
            load_setting( settings, ('engine', 'libpath') ), workdir, elems, position,
        )
        os.chdir(home)

        return dftbplus_instance

    elif load_setting( settings, ('engine', 'type') ) == 'pyscf':
        
        from interface_pyscf import pyscf_manager

        atoms = get_geom(load_setting(settings, 'geom_file'), return_format = 'pyscf')
        basis = load_setting(settings, 'ao_basis')
        unit  = 'Angstrom'

        pyscf_instance = pyscf_manager(atoms, basis, unit)

        return pyscf_instance

    else:

        stop_with_ettor( "Unknown quantum chemistry package %s ." % load_setting( settings, ('engine', 'type') ) )
