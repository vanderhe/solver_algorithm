#!/usr/bin/env python3
"""Routines for solving a linear system of equations."""
import numpy as np
import random_matrix

def gaussian_eliminate(aa, bb):
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        aa: Matrix with the coefficients. Shape: (n, n).
        bb: Right hand side of the equation. Shape: (n,)

    Returns:
        Vector xx with the solution of the linear equation or None
        if the equations are linearly dependent.
    """

    aa_rows = aa.shape[0]
    bb_rows = aa_rows
    aa_columns = aa.shape[1]

    #Test for linear dependency:
    det = np.linalg.det(aa)
    if abs(det) > 5*10**(-13):

        #Transformation to the lower-triangular-shape form:
        for ii in range(0, aa_rows, 1):
            for jj in range(ii + 1, aa_columns, 1):
                if aa[ii, ii] != 0:
                    var = (aa[jj, ii] / aa[ii, ii])
                    aa[jj, ii:] = aa[jj, ii:] - var * aa[ii, ii:]
                    bb[jj] = bb[jj] - var * bb[ii]
                elif aa[ii, ii] == 0:
                    switch_var = np.array(aa[ii, :])
                    aa[ii, :] = aa[jj, :]
                    aa[jj, :] = switch_var

        #Stack aa and bb to one matrix:
        bb_shape = bb.reshape((-1,1))
        sol = np.hstack([aa, bb_shape])

        for ii in range(0, aa_rows, 1):
            sol[ii, :] = sol[ii, :] / sol[ii, ii]

        print("lower-triangular-shape form:\n", sol)

        #Transformation to the final form:
        for ii in range(0, aa_rows, 1):
            for kk in range(ii + 1, aa_columns, 1):
                sol[ii, :] = sol[ii, :] - sol[ii, kk] * sol[kk, :]

    else:
        print("linear dependency detected!")
        sol = None

    print("\nfinal form:\n", sol)

    return sol

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
    
#Generate random matrix and "input_rand.txt":
random_matrix(3)

#Read matrices from file "input_rand.txt":
file_rand = open("input_rand.txt", "r")
lines_rand = file_rand.readlines()
numberOfVars_rand = len(lines_rand)
ma_in = np.ones((numberOfVars_rand, numberOfVars_rand +1))

#Fill Matrix ma_in with Numbers from input.txt:
for ii in range(0, numberOfVars_rand, 1):
    for jj in range(0, numberOfVars_rand +1, 1):
        ma_in[ii, jj] = float(lines_rand[ii].strip().split()[jj])
print("obtained matrix of input_rand.txt:\n", ma_in)

aa = ma_in[:, :numberOfVars_rand]
bb = ma_in[:, numberOfVars_rand]

##Read matrices from file "input.in":
#file = open("input.in", "r")
#lines = file.readlines()
##print("number of lines read:\n",lines)
#numberOfVars = int(lines[0].strip())
#aa = np.empty((numberOfVars, numberOfVars))
#bb_ma = np.empty((1,numberOfVars))
#
##Fill Matrix bb_ma with Numbers from input.in:
#lines_bb = lines[4].strip().split()
#for jj in range(0, numberOfVars, 1):
#        bb_ma[0, jj] = float(lines_bb[jj])
#bb = bb_ma[0, :]
#
##Fill Matrix aa with Numbers from input.in:
#for ii in range(1, numberOfVars +1, 1):
#    for jj in range(0, numberOfVars, 1):
#        aa[ii -1, jj] = float(lines[ii].strip().split()[jj])

print("obtained matrix aa:\n", aa)
print("\nobtained matrix bb:\n", bb, "\n")

#Solve linear equations system with method "gaussian_eliminate":
sol = gaussian_eliminate(aa, bb)

#Write obtained solution into file "output.out":
np.savetxt("output.out", sol)

#Compare with solution according numpy:
sol_numpy = np.linalg.solve(aa, bb)
print("\nnumpy solution:\n", sol_numpy)