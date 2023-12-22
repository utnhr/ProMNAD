#!-*- coding: utf-8 -*-

import numpy as np
import yaml

def read_input(filename):
    
    with open(filename, 'r') as input_file:
        
        try:
            input_obj = yaml.safe_load(input_file)
        except:
            utils.stop_with_error('Syntax error in input file.')

    return input_obj

def get_geom(filename, return_geom_1d=False):
    
    with open(filename, 'r') as geom_file:
        
        n_atom = int( geom_file.readline() )

        geom_file.readline() # skip comment line
        
        elems       = []
        elems_n_dof = []
        atoms_coord = []

        for i_atom in range(n_atom):
            
            ll = geom_file.readline().split()

            elem       = ll[0]
            atom_coord = [ float(ll[1]), float(ll[2]), float(ll[3]) ]
            
            elems.append(elem)
            elems_n_dof += [elem, elem, elem]
            atoms_coord.append(atom_coord)

    atoms_coord = np.array(atoms_coord)

    if return_geom_1d:

        #return elems_n_dof, atoms_coord.reshape(-1)
        return elems, atoms_coord.reshape(-1)

    else:
        
        return elems, atoms_coord

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
