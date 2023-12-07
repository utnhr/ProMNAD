#!-*- coding: utf-8 -*-

import constants
from tbfmodule import Tbf
from propagator import propagate

def do_dynamics():

    init_dynamics()
    
    for i_step in range(n_step):

        worlds = Tbf.worlds

        for world in worlds:
    
            world.propagate()

def init_dynamics():
    
    Tbf.
