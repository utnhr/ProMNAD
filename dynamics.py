#!-*- coding: utf-8 -*-

from constants import DEFAULT_DFTB_ANGMOM
from worldsmodule import World
from propagator import propagate
import utils
from inout import get_geom
from atomparammodule import Atomparam
import numpy as np

def do_dynamics(settings, is_restarted = False):

    init_dynamics(settings, is_restarted = is_restarted)

    utils.printer.write_out('Time propagation started.\n')
    
    for i_step in range(settings['n_step']):
        
        utils.printer.write_out("STEP %8d ...\n" % i_step)

        worlds = World.worlds

        for world in worlds:
    
            world.propagate()

    utils.printer.write_out('Time propagation finished.\n')


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

    return
