#!-*- coding: utf-8 -*-

import sys
import os
import glob
import subprocess
import numpy as np
import yaml
import utils
import ctypes

class dftbplus_manager:
    
    worker = None

    @classmethod
    def set_execution_environment(cls, working_directory=None, exe_path=None):

        if working_directory is not None:
        
            cls.working_directory = working_directory

        if exe_path is not None:
            
            cls.exe_path = exe_path

        return


    @classmethod
    def dict_to_dftbplus_input(cls, input_dict):
    
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
    
                    string += dict_to_dftbplus_input(value['values'])
    
                    string += "}\n"
    
                else:
                    
                    string += "%s = " % key
    
                    string += dict_to_dftbplus_input(value)
    
            else:
    
                if key == '(noname)':
                    
                    string += str(value) + "\n"
    
                elif key == 'is_named_dict':
                    
                    continue
    
                else:
                
                    string += "%s = %s\n" % ( key, str(value) )
    
        return string
    
    
    @classmethod
    def write_dftbplus_input(cls, template_filename, working_directory, elem_list, angmom_table):
        
        with open(template_filename, 'r') as template:
                
            input_dict = yaml.safe_load(template)
    
        dftb_block = input_dict['Hamiltonian']['DFTB']['values']
        
        dftb_block['MaxAngularMomentum'] = {}
        dftb_block['MaxAngularMomentum']['is_named_dict'] = False
        dftb_block['MaxAngularMomentum']['(noname)'] = {}
        dftb_block['MaxAngularMomentum']['(noname)']['is_named_dict'] = True
        dftb_block['MaxAngularMomentum']['(noname)']['values'] = {}
        dftb_block['MaxAngularMomentum']['(noname)']['values']['is_named_dict'] = False
        
        for elem in elem_list:
    
            dftb_block['MaxAngularMomentum']['(noname)']['values'][elem] = angmom_table[elem]
        
        string = dict_to_dftbplus_input(input_dict)
    
        with open(working_directory+'/dftb_in.hsd', 'w') as dftb_in:
            
            dftb_in.write(string)
    
    
    @classmethod
    def write_gen_format(cls, elems, geom, working_directory, is_1d=True, elem_list=None):
        
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
    
        with open(working_directory+'/geo.gen', 'w') as genfile:
            
            genfile.write("%d C\n" % n_atom)
    
            for elem in elem_list:
    
                genfile.write("%s " % elem)
                
            genfile.write("\n")
    
            if is_1d:
    
                for i_atom in range(n_atom):
    
                    genfile.write("%d %d %25.15f %25.15f %25.15f\n" % (
                            i_atom+1, elem_IDs[ elems[3*i_atom] ],
                            geom[3*i_atom], geom[3*i_atom+1], geom[3*i_atom+2]
                        )
                    )
    
            else:
                
                for i_atom in range(n_atom):
    
                    genfile.write("%d %d %25.15f %25.15f %25.15f\n" % (
                            i_atom+1, elem_IDs[ elems[i_atom] ],
                            geom[0, i_atom], geom[1, i_atom+1], geom[2, i_atom+2]
                        )
                    )
    
        return elem_list
    
    
    @classmethod
    def get_dftbplus_matrix_text(cls, working_directory, filename):
        
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
    
    
    @classmethod
    def run_dftbplus_text(cls, elems, angmom_table, geom, mode='construct_matrix'):
        
        if cls.working_directory is None:
            
            utils.stop_with_error('DFTB+ working directory is not specified.')

        if cls.exe_path is None:

            utils.stop_with_error('DFTB+ executable path is not specified.')
    
        elem_list = write_gen_format(elems, geom, working_directory)
    
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if mode == 'construct_matrix':
    
            template_filename = script_dir + '/input_templates/dftbplus_input_template_matrix.yaml'
    
        if mode == 'get_orbitals':
            
            template_filename = script_dir + '/input_templates/dftbplus_input_template_scc.yaml'
    
        write_dftbplus_input(template_filename, cls.working_directory, elem_list, angmom_table)
    
        job = subprocess.run([cls.exe_path], cwd = cls.working_directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        if mode == 'construct_matrix':
    
            n_AO, e_hamil = get_dftbplus_matrix_text(cls.working_directory, 'hamsqr1.dat')
            n_AO, overlap = get_dftbplus_matrix_text(cls.working_directory, 'oversqr.dat')
    
            return n_AO, e_hamil, overlap

    
    @classmethod
    #def dftbplus_init(cls, libpath, workdir, elems, angmom_table, geom):
    def dftbplus_init(cls, libpath, workdir):
        
        import dftbplus

        hsdpath = os.path.join(workdir, 'dftb_in.hsd')
        logpath = os.path.join(workdir, 'dftb.log')

        # make dftb_in.hsd

        #cls.worker = dftbplus.DftbPlus(libpath = libpath, hsdpath = hsdpath, logfile = logpath)
        cls.worker = dftbplus.DftbPlus(libpath = libpath, hsdpath = hsdpath)


