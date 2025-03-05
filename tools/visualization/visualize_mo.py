#!-*- coding: utf-8 -*-

import sys
import numpy as np
import re
import os
from pyscf import gto, scf, dft
from pyscf.tools import cubegen
sys.path.append( os.path.join( os.path.dirname(__file__), '..') )
import utils
import inout
from settingsmodule import load_setting

#inputfilename = 'atdkse.yaml'
inputfilename = sys.argv[1]
#mo_filename = 'mo_coeffs.0.dat'
mo_filename = sys.argv[2]
#is_real_mo = False
is_real_mo = bool(int(sys.argv[3])) # 0 (False) or 1 (True)
#traject_filename = 'traject.0.xyz'
traject_filename = sys.argv[4]
#i_mo = 10 # 0-based
i_mo = int(sys.argv[5])
#i_step = 1 # STEP No. indicated in traject.*.xyz
i_step = int(sys.argv[6]) # STEP No. indicated in traject.*.xyz
result_filename = sys.argv[7]
mesh_x = int(sys.argv[8])
mesh_y = int(sys.argv[9])
mesh_z = int(sys.argv[10])

# Read input file (to retrieve basis set information)
settings = inout.read_input(inputfilename)
basis = load_setting(settings, 'ao_basis')

# Read MO coeffs

mo_pattern = re.compile("^.*STEP\\s+%d\\s*.*$" % i_step)

mo_coeff = []

with open(mo_filename, 'r') as mo_file:
    
    while True:

        line = mo_file.readline()

        if mo_pattern.match(line):

            for i in range(i_mo):

                mo_file.readline()

            ll = mo_file.readline().split()

            if is_real_mo:
                
                for word in ll:

                    mo_coeff.append(float(word.rstrip(',')))
            
            else:

                for iword, word in enumerate(ll):

                    if iword % 2 == 0:
                        real_part = float(word.rstrip('+'))
                    else:
                        imag_part = float(word.rstrip('j,'))
                        c = real_part + (0.0+1.0j)*imag_part
                        mo_coeff.append(c)

            break

mo_coeff = np.array(mo_coeff)

# Read geometry

geom_pattern = re.compile("^.*STEP\\s*%d\\s*.*$" % i_step)

atoms = ""

with open(traject_filename, 'r') as traject_file:
    
    while True:

        n_atom = int(traject_file.readline())

        line = traject_file.readline()

        if geom_pattern.match(line):

            for i in range(n_atom):

                #ll = traject_file.readline().split()
                #elem = ll[0]
                #coord = np.array( [ float(ll[1]), float(ll[2]), float(ll[3]) ] )
                #atoms.append({'elem': elem, 'coord': coord})

                line = traject_file.readline()

                atoms += line

            break

        else:

            for i in range(n_atom):

                traject_file.readline()

# Construct 'density matrix' for MO under consideration
if is_real_mo:
    dm = np.outer(mo_coeff, mo_coeff)
else:
    dm = np.outer(mo_coeff.conjugate(), mo_coeff)

# PySCF initialization

mol = gto.Mole()
mol.unit = 'Angstrom'
mol.atom = atoms
mol.basis = basis
mol.symmetry = False
mol.build()

cubegen.density(mol, result_filename, dm=dm, nx=mesh_x, ny=mesh_y, nz=mesh_z)

