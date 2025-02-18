#!-*- coding: utf-8 -*-

import sys
import numpy as np
import re
import os
from copy import deepcopy
from pyscf import gto, scf, dft
from pyscf.tools import cubegen
sys.path.append( os.path.join( os.path.dirname(__file__), '..') )
import utils
import inout
from settingsmodule import load_setting

inputfilename = sys.argv[1]
mo_filename = sys.argv[2]
traject_filename = sys.argv[3]
is_real_mo = bool(int(sys.argv[4])) # 0 (False) or 1 (True)

# Read input file (to retrieve basis set information)
settings = inout.read_input(inputfilename)
basis = load_setting(settings, 'ao_basis')
dtau = load_setting(settings, 'dtau')

# Read MO coeffs

mo_steps = [] # indices of geometry steps in which MOs were written
mo_times = [] # times at which MOs were written
mo_coeffs = []

with open(mo_filename, 'r') as mo_file:
    
    while True:

        line = mo_file.readline()

        ll = line.split()
        
        try:
            step_id = int(ll[1])
        except:
            break

        mo_steps.append(step_id)
        mo_times.append( float(ll[3]) )

        n_read_mo = 0

        mo_coeff = []

        while True:

            n_read_mo += 1

            ll = mo_file.readline().split()

            if is_real_mo:
                n_ao = len(ll)
            else:
                n_ao = len(ll) // 2

            mo_vec = []

            if is_real_mo:
                
                for word in ll:

                    mo_vec.append( float( word.rstrip(',') ) )
            
            else:

                for iword, word in enumerate(ll):

                    if iword % 2 == 0:
                        real_part = float(word.rstrip('+'))
                    else:
                        imag_part = float(word.rstrip('j,'))
                        c = real_part + (0.0+1.0j)*imag_part
                        mo_vec.append(c)

            mo_vec = np.array(mo_vec)

            mo_coeff.append(mo_vec)

            n_mo = n_ao # assuming no basis-set truncation

            if n_read_mo == n_mo:
                break

        mo_coeffs.append( np.array(mo_coeff) )

# Read geometry

atom_positions = []

with open(traject_filename, 'r') as traject_file:

    for mo_step in mo_steps:
    
        geom_pattern = re.compile("^.*STEP\\s*%d\\s*.*$" % mo_step)
    
        atoms = ""
            
        while True:
        
            n_atom = int(traject_file.readline())
        
            line = traject_file.readline()
        
            if geom_pattern.match(line):
        
                for i in range(n_atom):
        
                    line = traject_file.readline()
        
                    atoms += line
    
                atom_positions.append(atoms)
    
                break
        
            else:
        
                for i in range(n_atom):
        
                    traject_file.readline()

n_snapshot = len(mo_steps)

for i_snapshot in range(1, n_snapshot):

    if i_snapshot == 1:
        mol_old = gto.Mole()
        mol_old.unit = 'Angstrom'
        mol_old.atom = atom_positions[i_snapshot-1]
        mol_old.basis = basis
        mol_old.symmetry = False
        mol_old.build()
    else:
        mol_old = deepcopy(mol_new)

    mol_new = gto.Mole()
    mol_new.unit = 'Angstrom'
    mol_new.atom = atom_positions[i_snapshot]
    mol_new.basis = basis
    mol_new.symmetry = False
    mol_new.build()

    overlap_twogeom = gto.intor_cross('int1e_ovlp', mol_old, mol_new)
    #overlap_twogeom_2 = gto.intor_cross('int1e_ovlp', mol_new, mol_old)

    #temp1 = np.triu(overlap_twogeom_1) + np.triu(overlap_twogeom_3).transpose()
    #temp2 = np.triu(overlap_twogeom_2) + np.triu(overlap_twogeom_4).transpose()

    #temp = temp1 - temp2

    #overlap_twogeom = temp

    timediff = dtau * (mo_steps[i_snapshot] - mo_steps[i_snapshot-1])

    mo_coeff_old = mo_coeffs[i_snapshot-1]
    mo_coeff_new = mo_coeffs[i_snapshot]

    mo_overlap = np.dot( mo_coeff_old.conjugate(), np.dot( overlap_twogeom, mo_coeff_new.transpose() ) )
    
    # correction for sign flipping

    for i_mo in range(n_mo):

        if mo_overlap[i_mo,i_mo] < 0:

            #sys.stderr.write("SIGN CORRECTED (%d, %20.12f)\n" % (i_mo, mo_overlap[i_mo,i_mo])) ## Debug code

            #mo_coeffs[i_snapshot][i_mo] *= -1.0
            mo_overlap[:,i_mo] = deepcopy(-mo_overlap[:,i_mo])

    #mo_coeff_new = mo_coeffs[i_snapshot]
    #mo_overlap = np.dot( mo_coeff_old.conjugate(), np.dot( overlap_twogeom, mo_coeff_new.transpose() ) )

    mo_tdnac = ( mo_overlap - mo_overlap.transpose() ) / timediff

    tdnac_step = (mo_steps[i_snapshot] + mo_steps[i_snapshot-1]) // 2
    tdnac_time = (mo_times[i_snapshot] + mo_times[i_snapshot-1]) / 2 # in fs

    mo_tdnac_file = sys.stdout

    time_str = "STEP %12d T= %20.12f fs\n" % (tdnac_step, tdnac_time)

    mo_tdnac_file.write(time_str)

    max_tdnac_norm = 0.0
    max_tdnac_mopair = None
    
    # Assume closed shell
    #for i_spin in range(n_spin):
    for i_mo in range(n_mo):

        for j_mo in range(n_mo):

            tdnac = mo_tdnac[i_mo,j_mo]

            mo_tdnac_file.write( "%20.12f+%20.12fj," % (tdnac.real, tdnac.imag) )

        mo_tdnac_file.write("\n")

