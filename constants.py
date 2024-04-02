#!-*- coding: utf-8 -*-

import math

# Physical and mathematical constants

H_PLANCK  = 2*math.pi # atomic units
H_DIRAC   = 1         # atomic units
E_MASS_KG = 9.1093837015e-31 # kg
AMU_KG    = 1.6605390666e-27 # kg
KB_EV     = 8.617333262e-5 # eV/K

# Unit converters

AMU2AU   = AMU_KG / E_MASS_KG
AU2ANGST = 0.529177210903
ANGST2AU = 1.0 / AU2ANGST
AU2SEC   = 2.4188843265857e-17
SEC2AU   = 1.0 / AU2SEC
AU2EV    = 27.211386245988

# Threshold

EPS_FLOAT_EQUAL = 1e-15

# atom mass table

ATOM_MASSES_AMU = {
    'H' :   1.00794 ,
    'He':   4.00260 ,
    'Li':   6.941   ,
    'Be':   9.012182,
    'B' :  10.811   ,
    'C' :  12.0107  ,
    'N' :  14.0067  ,
    'O' :  15.9994  ,
    'F' :  18.998403,
    'Ne':  20.1797  ,
    'Na':  22.989769,
    'Mg':  24.3050  ,
    'Al':  26.981539,
    'Si':  28.0855  ,
    'P' :  30.973762,
    'S' :  32.065   ,
    'Cl':  35.453   ,
}

# Default gaussian width table (1/Bohr^2)
# ref: Thompson, Punwong, and Martinez, Chem. Phys. 370, 70 (2010).

DEFAULT_GWP_WIDTH_AU = {
    'H' :  4.7,
    'C' : 22.7,
    'N' : 19.0,
    'O' : 12.2,
    'F' :  8.5,
    'S' : 16.7,
    'Cl':  7.4,
}

# Max. angular momenta for DFTB calculations

DEFAULT_DFTB_ANGMOM = {
    'H' : 's',
    'C' : 'p',
    'N' : 'p',
    'O' : 'p',
    'F' : 'p',
    'S' : 'd',
    'Cl': 'p',
}
