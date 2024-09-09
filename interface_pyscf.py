#!-*- coding: utf-8 -*-

from pyscf import gto, scf, dft

class pyscf_manager:
    
    def __init__(self, atoms, basis, unit):

        self.gto = gto
        self.dft = dft

        self.mol = gto.Mole()
        self.mol.unit = unit
        self.mol.atom = atoms
        self.mol.basis = basis
        self.mol.build()

