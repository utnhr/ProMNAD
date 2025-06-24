#!-*- coding: utf-8 -*-

import sys
import os
import glob
import subprocess
import numpy as np
import yaml
import utils
import ctypes
from constants import DEFAULT_DFTB_ANGMOM, AU2ANGST

class dftbplus_manager:

    
    def __init__(self, libpath, workdir, elem_list, geom, charge):
        
        import dftbplus

        self.worker            = None
        self.working_directory = None
        self.exe_path          = None
        self.template_filename = None
        self.angmom_table      = {}

        self.working_directory = workdir
        self.exe_path = libpath

        hsdpath = os.path.join(self.working_directory, 'dftb_in.hsd')
        logpath = os.path.join(self.working_directory, 'dftb.log')

        # make dftb_in.hsd
        self.write_gen_format(elem_list, geom, is_1d=True)
    
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.template_filename = script_dir + '/../dftbplus_input_templates/dftbplus_input_template_scc.yaml'

        for elem in elem_list:

            if elem not in self.angmom_table.keys():

                self.angmom_table[elem] = '"' + DEFAULT_DFTB_ANGMOM[elem] + '"'

        self.write_dftbplus_input(self.template_filename, elem_list, charge)

        self.worker = dftbplus.DftbPlus(libpath = libpath, hsdpath = hsdpath)

        return


    def go_to_workdir(self):

        self.home = os.getcwd()

        os.chdir(self.working_directory)

        return


    def return_from_workdir(self):
        
        os.chdir(self.home)

        return



    def dict_to_dftbplus_input(self, input_dict):
    
        if type(input_dict) is not dict:
    
            return str(input_dict)
    
        string = ''
    
        for item in input_dict.items():
            
            key   = item[0]
            value = item[1]
    
            if type(value) is dict:
    
                if 'is_named_dict' not in value.keys():
    
                    utils.stop_with_error("Cannot convert dictionary into string in dftb_in.hsd format: key 'is_named_dict' is missing'")
    
                if value['is_named_dict']:
                    
                    if key == '(noname)':
                        
                        string += " {\n"
    
                    else:
                        
                        string += "%s {\n" % key
    
                    string += self.dict_to_dftbplus_input(value['values'])
    
                    string += "}\n"
    
                else:
                    
                    string += "%s = " % key
    
                    string += self.dict_to_dftbplus_input(value)
    
            else:
    
                if key == '(noname)':
                    
                    string += str(value) + "\n"
    
                elif key == 'is_named_dict':
                    
                    continue
    
                else:
                
                    string += "%s = %s\n" % ( key, str(value) )
    
        return string
    
    
    def write_dftbplus_input(self, template_filename, elem_list, charge):
        
        with open(template_filename, 'r') as template:
                
            input_dict = yaml.safe_load(template)

        input_dict['Geometry']['GenFormat']['values']['(noname)'] = \
            '<<< ' + '"' + os.path.join(self.working_directory, 'geo.gen') + '"'

        dftb_block = input_dict['Hamiltonian']['DFTB']['values']
        
        dftb_block['MaxAngularMomentum'] = {}
        dftb_block['MaxAngularMomentum']['is_named_dict'] = False
        dftb_block['MaxAngularMomentum']['(noname)'] = {}
        dftb_block['MaxAngularMomentum']['(noname)']['is_named_dict'] = True
        dftb_block['MaxAngularMomentum']['(noname)']['values'] = {}
        dftb_block['MaxAngularMomentum']['(noname)']['values']['is_named_dict'] = False

        dftb_block['Charge'] = charge
        
        for elem in elem_list:
    
            dftb_block['MaxAngularMomentum']['(noname)']['values'][elem] = self.angmom_table[elem]

        dftb_block['SlaterKosterFiles']['Type2FileNames']['values']['Prefix'] = \
            self.working_directory + '/'
        
        string = self.dict_to_dftbplus_input(input_dict)
    
        with open( os.path.join(self.working_directory, 'dftb_in.hsd'), 'w' ) as dftb_in:
            
            dftb_in.write(string)

        return
    
    
    def write_gen_format(self, elems, geom, is_1d=True, elem_list=None):
        
        # geom can given in either two formats: [3*n_atom] array or [3, n_atom] array.
        # if is_1d is True, former is specified
    
        if is_1d:
    
            if len(elems) % 3 != 0:
    
                utils.stop_with_error("In 1D format, length of element list must be 3*N_ATOM.")
    
            n_atom = len(elems) // 3
    
        else:
    
            n_atom = len(elems)
    
        if elem_list is None:
    
            elem_list = []
    
            for i_atom in range(n_atom):
    
                elem = elems[i_atom + 2*i_atom*int(is_1d)]
    
                if elem not in elem_list:
    
                    elem_list.append(elem)
        
        # element -> element ID
    
        elem_IDs = {}
    
        for i_elem, elem in enumerate(elem_list):
            
            elem_IDs[elem] = i_elem + 1

        with open( os.path.join(self.working_directory, 'geo.gen'), 'w' ) as genfile:
            
            genfile.write("%d C\n" % n_atom)
    
            for elem in elem_list:
    
                genfile.write("%s " % elem)
                
            genfile.write("\n")
    
            if is_1d:
    
                for i_atom in range(n_atom):
    
                    genfile.write("%d %d %25.15f %25.15f %25.15f\n" % (
                            i_atom+1, elem_IDs[ elems[3*i_atom] ],
                            geom[3*i_atom]*AU2ANGST, geom[3*i_atom+1]*AU2ANGST, geom[3*i_atom+2]*AU2ANGST
                        )
                    )
    
            else:
                
                for i_atom in range(n_atom):
    
                    genfile.write("%d %d %25.15f %25.15f %25.15f\n" % (
                            i_atom+1, elem_IDs[ elems[i_atom] ],
                            geom[0, i_atom]*AU2ANGST, geom[1, i_atom+1]*AU2ANGST, geom[2, i_atom+2]*AU2ANGST
                        )
                    )
    
        return elem_list
    
    
    def get_dftbplus_matrix_text(self, working_directory, filename):
        
        with open(working_directory+'/'+filename, 'r') as matrix_file:
            
            matrix_file.readline() # skip 1 line
    
            n_AO = int( matrix_file.readline().split()[1] )
            
            # skip 3 lines
            for i in range(3):
                matrix_file.readline()
    
            matrix = []
    
            for i_AO in range(n_AO):
                
                # DFTB+ might store a 'column' as a 'row', but no problem because hamiltonian and overlap matrices are symmetric
                row = [ float(word) for word in matrix_file.readline().split() ]
    
                matrix.append(row)
    
            return n_AO, np.array(matrix)
    
    
    #@classmethod
    #def run_dftbplus_text(cls, atomparams, geom, mode = 'read_matrix'): # TODO

    #    if mode == 'read_matrix':
    #
    #        n_AO, e_hamil = cls.get_dftbplus_matrix_text(cls.working_directory, 'hamsqr1.dat')
    #        n_AO, overlap = cls.get_dftbplus_matrix_text(cls.working_directory, 'oversqr.dat')

    #        return n_AO, e_hamil, overlap
    #    
    #    if cls.working_directory is None:
    #        
    #        utils.stop_with_error('DFTB+ working directory is not specified.')

    #    if cls.worker is None:

    #        utils.stop_with_error('DFTB+ executable path is not specified.')
    #
    #    #elem_list = write_gen_format(elems, geom, working_directory)
    #
    #    #script_dir = os.path.dirname(os.path.abspath(__file__))

    #    #if mode == 'construct_matrix':
    #
    #    #    template_filename = script_dir + '/input_templates/dftbplus_input_template_matrix.yaml'
    #
    #    #if mode == 'get_orbitals':
    #    #    
    #    #    template_filename = script_dir + '/input_templates/dftbplus_input_template_scc.yaml'
    #
    #    #write_dftbplus_input(template_filename, elem_list, angmom_table)
    #
    #    #job = subprocess.run([cls.exe_path], cwd = cls.working_directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    #    return
