#!-*- coding: utf-8 -*-

from constants import DEFAULT_DFTB_ANGMOM
from worldsmodule import World
from propagator import propagate
from utils import stop_with_error
from inout import get_geom
from atomparammodule import Atomparam
import numpy as np

def do_dynamics(settings, is_restarted = False):

    init_dynamics(settings, is_restarted = is_restarted)
    
    for i_step in range(settings['n_step']):

        worlds = World.worlds

        for world in worlds:
    
            world.propagate()

def init_dynamics(settings, is_restarted = False):

    if is_restarted:
        
        stop_with_error("Restart not yet implemented.")

    else:

        elem, position = get_geom(settings['geom_file'], return_geom_1d = True)

        atomparams = []

        for iatom, elem in enumerate(elem):

            atomparams.append( Atomparam(elem, DEFAULT_DFTB_ANGMOM[elem]) )

        world = World(settings)
    
        world.set_initial_state( atomparams, position, np.zeros_like(position) )

