#!-*- coding: utf-8 -*-


class GlobalOutputFiles:
    """Output files for each world (common for all trajectories)"""

    def __init__(self, index): # index specifies the world
        
        self.tbf_coeff         = open("tbf_coeff.world%d.dat" % index, 'w') # TBF coefficients
        self.tbf_coeff_nophase = open("tbf_coeff_nophase.world%d.dat" % index, 'w') # TBF coefficients without phase factor
        self.tbf_popul         = open("tbf_popul.world%d.dat" % index, 'w') # TBF populations
        self.time              = open("time.world%d.dat" % index, 'w') # elapsed time from step 0
        
        return


    def __del__(self):
        
        self.tbf_coeff.close()
        self.tbf_coeff_nophase.close()
        self.tbf_popul.close()
        self.time.close()

        return


    def flush(self):
        
        self.tbf_coeff.flush()
        self.tbf_coeff_nophase.flush()
        self.tbf_popul.flush()
        self.time.flush()

        return


class LocalOutputFiles:
    """Output files for each trajectory."""   

    def __init__(self, index): # index specifies the trajectory i.e. TBF
        
        self.traject  = open("traject.%d.xyz" % index, 'w')   # geometry
        self.velocity = open("velocity.%d.xyz" % index, 'w')  # velocity
        self.energy   = open("energy.%d.dat" % index, 'w')    # potential en., kinetic en., temperature
        self.pec      = open("pec.%d.dat" % index, 'w')       # energies of each electronic state
        self.e_coeff  = open("e_coeff.%d.dat" % index, 'w')   # coefficients of electronic states
        self.e_popul  = open("e_popul.%d.dat" % index, 'w')   # populations of electronic states
        self.e_ortho  = open("e_ortho.%d.dat" % index, 'w')   # orthonormality of MOs, i.e., C^\dag*S*C
        self.mo_level = open("mo_level.%d.dat" % index, 'w')  # MO levels
        self.cmo_level= open("cmo_level.%d.dat" % index, 'w') # Canonical MO levels
        self.det      = open("det.%d.dat" % index, 'w')       # |S_occ| and |S_vir|
        self.mo_tdnac = open("mo_tdnac.%d.dat" % index, 'w')  # MO TDNAC
        self.mo_coeffs= open("mo_coeffs.%d.dat" % index, 'w') # MO coeffs (without phase)
        
        return


    def __del__(self):
        
        self.traject.close()
        self.velocity.close()
        self.energy.close()
        self.pec.close()
        self.e_coeff.close()
        self.e_popul.close()
        self.e_ortho.close()
        self.mo_level.close()
        self.cmo_level.close()
        self.det.close()
        self.mo_tdnac.close()
        self.mo_coeffs.close()

        return


    def flush(self):
        
        self.traject.flush()
        self.velocity.flush()
        self.energy.flush()
        self.pec.flush()
        self.e_coeff.flush()
        self.e_popul.flush()
        self.e_ortho.flush()
        self.mo_level.flush()
        self.cmo_level.flush()
        self.det.flush()
        self.mo_tdnac.flush()
        self.mo_coeffs.flush()
        
        return
