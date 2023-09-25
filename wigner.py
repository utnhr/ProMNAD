#!-*- coding: utf-8 -*-


import sys
import scipy.integrate as integrate
import numpy as np
import random
import math
import re


# Wigner distribution function for vibrational ground state
# Gaussian form of vib. wavefunction is assumed
class WignerDistribution:
    
    nsigma_limit = 10.0
    
    def __init__(self, sigma):
        
        self.wavefunc = \
            lambda x: ( 1.0 / (math.pi * sigma**2) )**0.25 * \
                      math.exp( - x**2 / (2.0 * sigma**2) )

        self.sigma = sigma

    def get_prob_density(self, x, p):
        
        integ, err = integrate.quad(
            lambda y: self.wavefunc(x+y)*self.wavefunc(x-y)*math.cos(2.0*p*y),
            -(WignerDistribution.nsigma_limit * self.sigma + x),
             (WignerDistribution.nsigma_limit * self.sigma - x),
        ) # atomic units (hbar = 1)

        prob_density = integ / math.pi # atomic units (hbar = 1)

        return prob_density


# unit conversion constants
au_in_amu    = 9.1093837015e-31 / 1.66053906660e-27 # mass
au_in_angst  = 0.529177210903 # distance
au_in_sec    = 2.418884326502e-17 # time
kayser_to_au = 2.0*math.pi             * au_in_sec             * 299792458.0  * 100.0
#              freq. -> angular freq.  s^-1 -> a.u.(time)^-1   m^-1 -> s^-1   cm^-1 -> m^-1


# element symbols
elem_symbols = [
    "DUMMY",
    "H" ,                                                                                                 "He",
    "Li", "Be",                                                             "B" , "C" , "N" , "O" , "F" , "Ne",
    "Na", "Mg",                                                             "Al", "Si", "P" , "S" , "Cl", "Ar",
    "K" , "Ca", "Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I" , "Xe",
    "Cs", "Ba",
]
                


# set sampling parameters
nsample               = 501
nsigma_max            = 4.0
random.seed(20191113)


# read dftbk_vib.out file
standard_orientation_pattern = re.compile("^\s*Standard orientation:")
freq_pattern                 = re.compile("^\s*Frequencies --")
horizontal_bar_pattern       = re.compile("^\s*--------------------------------------------------------------------------------")

vib_output = sys.stdin

vib_modes = []
atoms     = []

sys.stdout.write("READING ...\n")

for line in vib_output:
    
    # determine the number of atoms & read geometry
    if standard_orientation_pattern.match(line):

        for i in range(4):
            vib_output.readline()

        while True:
            line = vib_output.readline()
            if not horizontal_bar_pattern.match(line):
                ll = line.split()
                atoms.append(
                    {   
                        "atomic_number": int(ll[1]),
                        "atomic_type"  : int(ll[2]),
                        "coord"        : np.array(
                            [ float(ll[3]), float(ll[4]), float(ll[5]) ]
                        ) / au_in_angst,
                    }
                )
            else:
                break

        natom = len(atoms)
    
    # read vibrational modes
    if freq_pattern.match(line):

        ll_freq = line.split()

        ll_mass = vib_output.readline().split()

        lls_deform_vec = []
        for i in range(5):
            vib_output.readline()
        for iatom in range(natom):
            lls_deform_vec.append( vib_output.readline().split() )
        
        nmode_in_this_line = 0
        
        # read frequency
        for word in ll_freq[2:]:
            nmode_in_this_line += 1

            vib_modes.append(
                {
                    "frequency": float(word) * kayser_to_au # a.u. (angular frequency)
                }
            )

        # read mass
        for iword, word in enumerate(ll_mass[3:nmode_in_this_line+3]):
            vib_modes[ -nmode_in_this_line + iword ]["mass"] = \
                float(word) / au_in_amu # a.u.

        # read deformation vector
        for imode_in_this_line in range(nmode_in_this_line):
            vib_modes[
                -nmode_in_this_line + imode_in_this_line
            ]["vector"] = [ None for i in range(3*natom) ]
                
            for iatom, ll_deform_vec in enumerate(lls_deform_vec):

                count = 0
                for iword in range( 2+imode_in_this_line*3, 2+(imode_in_this_line+1)*3 ):

                    vib_modes[ 
                        -nmode_in_this_line + imode_in_this_line
                    ]["vector"][3*iatom+count] = float( ll_deform_vec[iword] )
                                        
                    count += 1

        # normalization
        for imode, mode in enumerate(vib_modes):
            mode["vector"] = np.array( mode["vector"] )
            vib_modes[imode]["vector"] = mode["vector"] / np.linalg.norm( mode["vector"] )
 

# generate Wigner distribution for each mode
samples = [
    [ None for imode in range( len(vib_modes) ) ] for isample in range(nsample)
]

for imode, mode in enumerate(vib_modes):
    
    sys.stderr.write("GENERATING ENSEMBLE FOR MODE %d ...\n" % imode)
    
    b = math.sqrt(
        1.0 / ( mode["mass"] * mode["frequency"] )
    )
    
    W = WignerDistribution(b)
    
    x_abs_max = nsigma_max * b
    p_abs_max = nsigma_max * (1.0/b) # momentum representation

    prob_density_max = W.get_prob_density(0.0, 0.0)
    
    # von Neumann 
    isample = 0

    while isample < nsample:

        x = random.uniform( -x_abs_max, x_abs_max )
        p = random.uniform( -p_abs_max, p_abs_max )

        prob_density = W.get_prob_density(x, p)

        if prob_density > random.uniform(0.0, prob_density_max):
            
            samples[isample][imode] = (x, p)

            isample += 1


# output resulting geometries and velocities
for isample, sample in enumerate(samples):

    geom_output_filename  = "%d.geom.xyz" % isample
    veloc_output_filename = "%d.veloc.dat" % isample
    
    # generate deformed geometry
    new_geom = []
    for atom in atoms:
        new_geom.append( atom["coord"] * au_in_angst ) # in Angstrom

    for mode, xp_pair in zip(vib_modes, sample):

        deform = mode["vector"] * xp_pair[0] * au_in_angst # in Angstrom
        
        for iatom, atom in enumerate(atoms):

            new_geom[iatom] += deform[3*iatom : 3*iatom+3] # in Angstrom
        
    # output of geometry
    with open(geom_output_filename, "w") as geom_output:

        geom_output.write("%d\n" % natom)
        geom_output.write("Wigner\n")

        for atom, coord in zip(atoms, new_geom):
            
            elem_symbol = elem_symbols[ atom["atomic_number"] ]

            geom_output.write(
                "%s %20.12f %20.12f %20.12f\n" % (
                    elem_symbol, coord[0], coord[1], coord[2]
                )
            )

    # calculate & output velocity
    # in Angstrom / fs
    veloc = np.array( [ 0.0 for i in range(3*natom) ] )
    for mode, xp_pair in zip(vib_modes, sample):
        veloc += mode["vector"] * ( xp_pair[1] / mode["mass"] ) \
            * au_in_angst / (au_in_sec/1.0e-15)
    
    with open(veloc_output_filename, "w") as veloc_output:
        for iatom in range(natom):
            veloc_output.write(
                "%20.12f %20.12f %20.12f\n" % (
                    veloc[3*iatom+0], veloc[3*iatom+1], veloc[3*iatom+2]
                )
            )
        
