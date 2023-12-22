#!-*- coding: utf-8 -*-

class Settings:
    
    def __init__(self):
        
        self.load_default_settings()

    def load_default_settings(self): # placeholder

        self.qc_program       = None

        self.n_occ            = None
        self.active_orbitals  = None
        
        self.n_estate         = None

        self.dt               = None
        self.propagator_type  = None
        
        return 
