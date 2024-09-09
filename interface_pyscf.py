#!-*- coding: utf-8 -*-

from settingsmodule import load_setting
from copy import deepcopy
from pyscf import gto, scf, dft

class pyscf_manager:
    
    def __init__(self, settings, atoms):

        self.gto = gto
        self.dft = dft

        self.mol = gto.Mole()
        self.mol.unit = 'Angstrom'
        self.mol.atom = atoms
        self.mol.basis = load_setting(settings, 'ao_basis')
        self.mol.symmetry = False
        self.mol.build()

        self.ks = self.dft.KS(self.mol) # currently only RKS is supprted (mol.spin==0 assumed)
        self.ks.xc = load_setting(settings, 'xc')
        
        return


    def update_geometry(self, coords):
        
        atoms = deepcopy(self.mol._atom)

        elems = [ atom[0] for atom in atoms ]

        new_atoms = [ [ elem, coord ] for elem, coord in zip(elems, coords) ]

        self.mol.atom = new_atoms

        self.mol.build()

        return


    def converge_scf(self):
        
        self.ks.kernel()

        return
