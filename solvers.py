"""Routines for solving a linear system of equations."""
import numpy as np


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
    
    #print("lower-triangular-shape form:\n", sol)
    
    #Transformation to the final form:        
    for ii in range(0, aa_rows, 1):
        for kk in range(ii + 1, aa_columns, 1):
            sol[ii, :] = sol[ii, :] - sol[ii, kk] * sol[kk, :]
    sol_bb = sol[:, aa_columns]
    
    #print("\nfinal form:\n", sol)    
    
    return sol_bb