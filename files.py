#!-*- coding: utf-8 -*-


class GlobalOutputFiles:
    """Output files for entire job (common for all trajectories)"""

    @classmethod
    def initialize(cls):
        
        cls.tbf_coeff = open('tbf_coeff.dat', 'w') # TBF coefficients
        cls.tbf_popul = open('tbf_popul.dat', 'w') # TBF populations
        
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
        
        return


    def __del__(self):
        
        self.traject.close()
        self.velocity.close()
        self.energy.close()
        self.pec.close()
        self.e_coeff.close()
        self.e_ortho.close()

        return
