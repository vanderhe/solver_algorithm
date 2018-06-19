#!/usr/bin/env python3
"""Routines for generating a random linear system of equations."""

import numpy as np

def random_matrix(numVars):
    """Generates a random linear system of equations

    Args:
        numVars: Number of variables, which defines the shape

    Returns:
        Random matrix rand_ma 
    """
    
    #Generating matrices using numpy routines:
    aa_rand = np.random.rand(numVars, numVars)
    bb_rand = np.random.rand(numVars, 1)

    #Stack aa and bb to one matrix:
    ma_rand = np.hstack([aa_rand, bb_rand])    
    
    print("obtained random matrix:\n", ma_rand)
    
    #Write obtained matrix into file "input_rand.txt":
    np.savetxt("input_rand.txt", ma_rand)
    
    return ma_rand
