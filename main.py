#!-*- coding: utf-8 -*-

import sys
import os
import numpy as np

import tbfmodule
import constants
import utils
import inout
import sample
from interface_dftbplus import dftbplus_manager
import electronmodule

if __name__ == '__main__':

    input_filename = sys.argv[1]

    settings = inout.read_input(input_filename)

    if settings['mode'] == 'test':

        elems, geom   = inout.get_geom('sample_geom.xyz', return_1d = True)

        n_AO, hamil   = inout.get_matrix_from_text('sample_hamil.dat')
        n_AO, overlap = inout.get_matrix_from_text('sample_overlap.dat')

        angmom_table = { 'C': '"p"', 'H': '"s"', 'O': '"p"' }
        exe_path     = '/home/hiroki/dftbplus-devel/niehaus/dftbplus/bin/dftb+'

        atominfo = { 'elems': elems, 'angmom_table': angmom_table }

        dftbplus_manager.set_execution_environment(workdir = os.getcwd()+'/dftbplus_workdir', exe_path = exe_path)

        print(elems)

        test1 = tbfmodule.tbf(position = geom, n_dof = 3 * len(geom))
        test2 = tbfmodule.tbf(position = geom, n_dof = 3 * len(geom))
        test3 = tbfmodule.tbf(position = geom, n_dof = 3 * len(geom))

        print(tbfmodule.tbf.live_tbfs)
        print(tbfmodule.tbf.live_tbf_count)
        print(tbfmodule.tbf.total_tbf_count)

        test1.destroy()

        print(tbfmodule.tbf.live_tbfs)
        print(tbfmodule.tbf.live_tbf_count)
        print(tbfmodule.tbf.total_tbf_count)
        
        n_AO, H, S = dftbplus_manager.run_dftbplus_text(atominfo, geom)

        mo_energies, mo_coeffs = electronmodule.get_molecular_orbitals(H, S)

        print(mo_energies, mo_coeffs)

        print(mo_coeffs[0,:])

    else:
        
        utils.stop_with_error("Mode %s not implemented." % settings['mode'])
