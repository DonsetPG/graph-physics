import os
import numpy as np
import warnings

import torch


# =========================================================================================
# Linear algebra stuff: solving linear systems and Steihaug-Toint Conjugate Gradient method
# =========================================================================================


def eye_like(A):    # Returns torch identity matrix with same dimensions, data type, and device as A
    assert(A.ndim == 2 and A.shape[0] == A.shape[1])
    if not isinstance(A, torch.Tensor): 
        return np.eye(A.shape[0])
    else:
        return torch.eye(A.shape[0], dtype=A.dtype, device=A.device)


def solve_linear_psd(A, b, method=None):
    # Solve linear system Ax = b. We assume that A is symmetric and positive semi-definite
    # eps is used to add a small value to the diagonal of A, for numerical stability

    if not isinstance(A, torch.Tensor): 
        if method is None:
            method = 'solve'
        try:
            do_lstsq = False
            if method == 'lstsq':
                do_lstsq = True
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            elif method == 'solve':
                x = np.linalg.solve(A, b)
            else:
                raise Exception("Method must be 'solve' or 'lstsq' if A is not a torch tensor")

        except RuntimeError as e:
            # If other methods fail (e.g., A is only PSD but not strictly positive definite),
            # fall back to a more robust method
            print(f"Warning: numpy got error {str(e)} with method np.linalg.solve, using lstsq instead")
            do_lstsq = True

        if do_lstsq:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    else:  # torch tensor
        if method is None:
            method = 'solve_ex'
            
        do_lstsq = False  # Fallback

        try:
            if method=='solve':
                x = torch.linalg.solve(A, b)

            elif method=='solve_ex':
                x = torch.linalg.solve_ex(A, b)[0]

            elif method == 'cholesky':
                L = torch.linalg.cholesky(A)
                x = torch.cholesky_solve(b.unsqueeze(-1), L, upper=False)
                
                return  x.squeeze()

            elif method == 'cholesky_ex':
                L = torch.linalg.cholesky_ex(A)[0]
                x = torch.cholesky_solve(b.unsqueeze(-1), L, upper=False)
                
                return  x.squeeze()

            elif method == 'QR':
                Q, R = torch.linalg.qr(A)
                x = torch.linalg.solve_triangular( R, (Q.T @ b).unsqueeze(1), upper=True).squeeze()

            elif method=='inv':
                x = torch.linalg.inv(A)@b

            elif method=='lstsq':
                do_lstsq = True

            else:
                assert False, f"Unknown method {method} in solve_linear_psd"

        except RuntimeError as e:
            if isinstance(e, NotImplementedError):
                raise        
            # If other methods fail (e.g., A is only PSD but not strictly positive definite),
            # fall back to a more robust method
            print(f"Warning: torch got error {str(e)} with method {method}, using lstsq instead")
            do_lstsq = True

        if do_lstsq:
            x = torch.linalg.lstsq(A, b).solution

    return x




def benchmark_linsolve(num_runs=10, printx=False):
    # This function is used to benchmark the linear solvers
    # It creates a random symmetric positive definite matrix and a random vector
    # Then it solves the linear system Ax = b using different methods
    # Finally, it prints the time taken for each method
    # It can be run as
    #   > python -c 'import linear_solvers; linear_solvers.benchmark_linsolve()'
    #
    # Arguments
    #   num_runs: Number of runs to average the time taken for each method
    #   printx: If True, print the solution for each method

    import time
    import utils
    def get_A_b():
        n = 1000
        rand_mat = torch.randn(n, n)
        A = rand_mat @ rand_mat.T  # This creates a symmetric PSD matrix
        A += eye_like(A)*1e-4
        b = torch.randn(n)
        return A, b

    for method in ['solve','solve_ex','cholesky','cholesky_ex','QR','lstsq','inv']:
        tot_time = 0
        if printx:
            print()
        for i in range(num_runs):
            torch.manual_seed(i)
            A, b = get_A_b()
            stime = time.time()
            x = solve_linear_psd(A, b, method=method)
            tot_time += time.time() - stime
            utils.empty_torch_cache()
            if printx:
                print(f'{method:15s} {x[:4].cpu().numpy()}')
        print(f'{method:15s} {tot_time:3f}')
