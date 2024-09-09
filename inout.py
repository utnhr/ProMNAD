#!-*- coding: utf-8 -*-

import numpy as np
import yaml
import utils
from constants import ANGST2AU, SEC2AU

def read_input(filename):
    
    with open(filename, 'r') as input_file:
        
        try:
            input_obj = yaml.safe_load(input_file)
        except:
            utils.stop_with_error('Syntax error in input file.')

    return input_obj

def get_geom(filename, return_geom_1d=False, return_elems_1d=False, return_format = None):

    with open(filename, 'r') as geom_file:
        
        n_atom = int( geom_file.readline() )

        geom_file.readline() # skip comment line
        
        elems       = []
        elems_n_dof = []
        atoms_coord = []

        for i_atom in range(n_atom):
            
            ll = geom_file.readline().split()

            elem       = ll[0]
            atom_coord = np.array( [ float(ll[1]), float(ll[2]), float(ll[3]) ] ) * ANGST2AU
            
            elems.append(elem)
            elems_n_dof += [elem, elem, elem]
            atoms_coord.append(atom_coord)

    atoms_coord = np.array(atoms_coord)

    if return_elems_1d:

        elems = elems_n_dof

    if return_geom_1d:

        return elems, atoms_coord.reshape(-1)

    if return_format == 'pyscf':

        return utils.coord_1d_to_pyscf(atoms_coord.reshape(-1), elems)

    else:
        
        return elems, atoms_coord

def get_traject(filename_geom, filename_veloc, return_geom_1d=False):

    filenames = [ filename_geom, filename_veloc ]

    geom = []; veloc = []

    for i, filename in enumerate(filenames):

        with open(filename, 'r') as traject_file:
            
            while True:

                line = traject_file.readline()

                if not line:
                    break

                n_atom = int(line)

                traject_file.readline() # skip 1 line

                vect = []

                for i_atom in range(n_atom):

                    ll = traject_file.readline().split()
                    
                    coord = np.array( [ float(ll[1]), float(ll[2]), float(ll[3]) ] )

                    if i == 0: # geometry; Angst -> a.u.

                        coord *= ANGST2AU

                    elif i == 1: # velocity; Angst/fs -> a.u.

                        coord *= ANGST2AU / (SEC2AU * 1.0e-15)

                    vect.append(coord)

                vect = np.array(vect)

                if return_geom_1d:

                    vect = vect.reshape(-1)

                if i == 0:

                    geom.append(vect)

                elif i == 1:
                    
                    veloc.append(vect)

    return geom, veloc

def get_matrix_from_text(filename='sample_hamil.dat'):
    
    with open(filename, 'r') as matrix_file:
        
        n_AO = int( matrix_file.readline() )

        matrix = []

        for i_AO in range(n_AO):

            ll = matrix_file.readline().split()

            row = [ float(ll[i_word]) for i_word in range(n_AO) ]

            matrix.append(row)

    matrix = np.array(matrix)

    return n_AO, matrix
