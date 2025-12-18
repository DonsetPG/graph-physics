import time
from collections import namedtuple
from collections.abc import Iterable
import numpy as np
from . import linear_solvers
# A few useful functions. These work with both torch tensors and numpy arrays
def is_infnan(x): # return True if x is either infinite or NaN
    x = float(x)
    return np.isinf(x) or np.isnan(x)

def l1norm(x):    # Used when printing debugging information
    return float(max(x.max(), -x.min())) if x is not None else 0



# This is the Objective class that should be implemented by the user, and it should 
# provide the methods get_objective, get_gradient, and get_hessian (if needed by the optimizer). 
# Instances can be passed in as optimize(..., optimizer=objective_instance) 
class Objective(object):
    def initialize_parameters(self, x0):  # Format initial parameter value x0 (e.g., as torch tensor if needed)
        return x0
    def get_objective(self, x): # Return objective value for parameters x
        raise NotImplementedError
    def get_gradient(self, x): # Return gradient of the objective function for parameters x
        raise NotImplementedError 
    def get_hessian(self, x):  # Return hessian of the objective function for parameters x
        raise NotImplementedError 
    


# This is the base class for all optimizers. It provides a common interface for all optimizers
# and defines the methods that should be implemented by each specific optimizer.
class Optimizer(object):
    # Default parameters and parameter values for the optimizer
    default_max_iter = 1000  # default maximum number of iterations
    minimize = True  # whether to minimize or maximize the objective function

    def reset_state(self): # Reset the optimizer internal state if needed
        pass

    def __init__(self, patience=None, tol=None, minimize=None, verbose=False):
        # Initialize optimizer with universally-shared parameters
        # Arguments:
        #   minimize (bool) : whether to minimize or maximize the objective function
        #   max_iter (int)  : maximum number of iterations
        #   patience (int)  : number of iterations to wait for validation improvement before stopping
        #   tol (float)     : early stopping once objective does not improve by more than tol
        #   minimize (bool) : whether to minimize or maximize the objective function
        #   verbose (int)   : verbosity level
        self.verbose  = verbose
        if minimize is not None:
            self.minimize = minimize
        self.tol      = tol      if tol is not None else 1e-8

        self.reset_state()  # Reset the optimizer state


    def msg(self, s, t=None):
        print(f"{self.__class__.__name__} : " + (f"iter={t:4d} :" if t is not None else "") + s)

    def get_update(self, t, objective, x):
        # This method provides the specific update logic for each optimizer, returning 
        # the updated parameters x_t+1, based on the current iteration t, objective object, 
        # and current parameters x
        #
        # IMPORTANT: this method can return either
        #   1. new_x (the updated parameters)
        #   2. (new_x, f_new_trn) (the updated parameters and the new training objective)
        # The logic in optimize(...) will handle both cases
        raise NotImplementedError


class GradientDescent(Optimizer):
    lr = 0.001
    default_max_iter = 10000
    def __init__(self, lr=None, **kwargs):
        # Gradient ascent optimizer. lr is the learning rate
        if lr is not None:
            self.lr = lr
        super().__init__(**kwargs)

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
        if self.minimize: grad = -grad
        return x + self.lr * grad
    

class GradientDescentBB(GradientDescent):
    # Gradient ascent with Barzilai-Borwein step sizes (short-step version)
    # TODO: check if optimizer can be unstable if lr is too large (and maybe even if too low), 
    # this can be tested using example_general.py

    default_max_iter = 1000

    def reset_state(self):
        # Reset the optimizer state
        self.previous_grad = None
        self.previous_x    = None

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
        if self.previous_grad is not None and self.previous_x is not None:
            last_delta_x = x    - self.previous_x
            d_grad       = grad - self.previous_grad
            
            # BB short step size
            alpha        = (last_delta_x @ d_grad) / (d_grad @ d_grad)

            if is_infnan(alpha):
                # If alpha is NaN, fall back to the default learning rate
                if self.verbose: print(f'GradientDescentBB : Invalid alpha!')
                alpha = self.lr
        else:
            alpha = self.lr if self.minimize else -self.lr

        self.previous_grad = grad
        self.previous_x    = x

        # Like with Newton's method, BB update is actually the same for minimization and maximization
        return x - alpha * grad  


class Adam(GradientDescent):
    # Gradient-based optimized using Adam (Adaptive Moment Estimation) optimizer by Kingma and Ba

    beta1 = 0.9   # exponential decay rate for the first moment estimates
    beta2 = 0.999 # exponential decay rate for the second moment estimates
    skip_warm_up = False # if True, we skip the warm-up phase for the first moment estimates
    eps = 1e-8    # small constant to prevent division by zero
    def __init__(self, beta1=0.9, beta2=0.999, skip_warm_up=False, eps=1e-8, **kwargs):
        if beta1 is not None:
            self.beta1 = beta1
        if beta2 is not None:
            self.beta2 = beta2
        if skip_warm_up is not None:
            self.skip_warm_up = skip_warm_up
        if eps is not None:
            self.eps = eps
        super().__init__(**kwargs)


    def reset_state(self): # Reset the optimizer state
        self.m = None
        self.v = None

    def get_update(self, t, objective, x):
        if self.m is None or self.v is None:
            self.m = x * 0
            self.v = x * 0

        grad = objective.get_gradient(x)
        if self.minimize: grad = -grad

        # Adam moment updates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        if self.skip_warm_up:
            m_hat = self.m
            v_hat = self.v
        else:
            m_hat = self.m / (1 - self.beta1 ** (t+1))
            v_hat = self.v / (1 - self.beta2 ** (t+1))

        # Compute parameter update
        delta_x = self.lr * m_hat / (v_hat**(1/2) + self.eps)
        return x + delta_x
    

class NewtonMethod(Optimizer):
    # Newton-Raphson method optimizer

    linsolve_eps = 1e-8  # regularization parameter for linear solvers (helps numerical stability)
    default_max_iter = 100   # maximum number of iterations

    def __init__(self, linsolve_eps=None, **kwargs):
        if linsolve_eps is not None:
            self.linsolve_eps = linsolve_eps
        super().__init__(**kwargs)

    def get_regularized_hessian(self, objective, x):
        # Get the Hessian matrix and add a small regularization term
        H = objective.get_hessian(x)
        if is_infnan(H.sum()):   # Error occured, usually it means x is too big
            if self.verbose: print(f'NewtonMethod : [Stopping] Invalid Hessian')
            return None 
        return H + self.linsolve_eps * linear_solvers.eye_like(H)

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
            
        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None
        
        return x - linear_solvers.solve_linear_psd(A=H_reg, b=grad) 



class NewtonMethodTrustRegion(NewtonMethod):
    # Newton's method with trust region constraint
    default_max_iter=1000
    trust_radius=1                # maximum trust region
    steihaug_toint_cg_tol=1e-10   # tolerance for Steihaug-Toint CG method
    def __init__(self, trust_radius=None, steihaug_toint_cg_tol=None, **kwargs):
        if trust_radius is not None:
            self.trust_radius = trust_radius
        if steihaug_toint_cg_tol is not None:
            self.steihaug_toint_cg_tol = steihaug_toint_cg_tol
        super().__init__(**kwargs)


    @staticmethod
    def steihaug_toint_cg(A, b, trust_radius, tol=1e-10, max_iter=None):
        """
        Steihaug-Toint Conjugate Gradient method for approximately solving
        argmin_x 0.5 x^T A x - b^T x  subject to ||x|| <= trust_radius
        where A is symmetric (not necessarily positive definite).
        
        Args:
            A : Symmetric matrix (n x n).
            b : Right-hand side vector (n).
            trust_radius (float): Trust region radius.
            tol (float): Tolerance for convergence on residual norm
            max_iter (int): Maximum number of iterations.
        
        Returns:
            torch.Tensor: Approximate solution x.
        """
        def find_tau(x, d, trust_radius):
            # Solve quadratic equation for tau: ||x + tau*d||^2 = trust_radius^2
            a = d @ d
            b_lin = 2 * (x @ d)
            c = (x @ x) - trust_radius**2
            discriminant = b_lin**2 - 4 * a * c
            sqrt_discriminant = discriminant**(1/2) if discriminant > 0 else 0
            tau = (-b_lin + sqrt_discriminant) / (2 * a)
            return tau

        n = b.shape[0]
        max_iter = max_iter or 10 * n
        x = b * 0      # initialize to 0s
        r = b + 0      # make a copy
        d = r + 0      # make a copy
        tol_sq          = tol**2
        trust_radius_sq = trust_radius**2

        if r@r < tol_sq:
            return x

        for _ in range(max_iter):
            Hd = A @ d
            dHd = d @ Hd

            if dHd <= 0: # Negative curvature detected → move to boundary
                tau = find_tau(x, d, trust_radius)
                return x + tau * d

            alpha = (r @ r) / dHd
            x_next = x + alpha * d

            if x_next@x_next >= trust_radius_sq: # Exceeding trust region → project onto boundary
                tau = find_tau(x, d, trust_radius)
                return x + tau * d

            r_next = r - alpha * Hd

            if r_next@r_next < tol_sq: # Converged
                return x_next

            beta = (r_next @ r_next) / (r @ r)
            d = r_next + beta * d

            x = x_next
            r = r_next

        return x
        

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)

        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None 

        if not self.minimize:
            grad, H_reg = -grad, -H_reg

        # Solve the constrained trust region problem using the
        # Steihaug-Toint Truncated Conjugate-Gradient Method
        delta_x = self.steihaug_toint_cg(A=H_reg, b=grad, trust_radius=self.trust_radius, tol=self.steihaug_toint_cg_tol)
        
        return x - delta_x

        
class TRON(NewtonMethodTrustRegion):
    # TRON method, Newton's method with adaptive trust region constraints
    # Reference:
    #   Lin and Moré, Newton's method for large bound-constrained optimization problems
    #   SIAM Journal on Optimization 9 (4), 1100-1127

    eta0 = 0.0   # Hyperparameters for adjusting trust radius
    eta1 = 0.25
    eta2 = 0.75
    trust_radius_min = 1e-3
    trust_radius_max = 1000.0
    trust_radius_adjust_max_iter = 100

    def __init__(self, eta0=None, eta1=None, eta2=None, trust_radius_min=None, trust_radius_max=None, trust_radius_adjust_max_iter=None, **kwargs):
        if eta0 is not None:
            self.eta0 = eta0
        if eta1 is not None:
            self.eta1 = eta1
        if eta2 is not None:
            self.eta2 = eta2
        if trust_radius_min is not None:
            self.trust_radius_min = trust_radius_min
        if trust_radius_max is not None:
            self.trust_radius_max = trust_radius_max
        if trust_radius_adjust_max_iter is not None:
            self.trust_radius_adjust_max_iter = trust_radius_adjust_max_iter

        super().__init__(**kwargs)


    def reset_state(self):
        self.adjusted_trust_radius = float(self.trust_radius)
        self.f_last_trn = None   # keep track of the last training objective


    def get_update(self, t, objective, x):
        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None

        if self.adjusted_trust_radius <= self.trust_radius_min:
            return None  # stop if we are at the minimum trust region

        grad = objective.get_gradient(x)
        if not self.minimize:
            grad, H_reg = -grad, -H_reg

        if not hasattr(self, 'f_last_trn') or self.f_last_trn is None:
            self.f_last_trn = objective.get_objective(x)
            if is_infnan(self.f_last_trn):
                if self.verbose: self.msg(f"[Stopping] Invalid training objective {self.f_last_trn}", t)
                return None
            
        for tr_iter in range(self.trust_radius_adjust_max_iter): # loop until adjusted_trust_radius is adjusted properly
            delta_x    = self.steihaug_toint_cg(A=H_reg, b=grad, tol=self.steihaug_toint_cg_tol, trust_radius=self.adjusted_trust_radius)

            new_x      = x - delta_x
            f_new_trn  = objective.get_objective(new_x)

            pred_improvement = (grad @ delta_x + 0.5 * delta_x @ (H_reg @ delta_x))
            improvement      = f_new_trn - self.f_last_trn
            if self.minimize: 
                improvement  = -improvement
            if self.verbose > 1: self.msg(f"pred_improvement={pred_improvement:.3f} improvement={improvement:.3f}", t)

            rho = improvement / (pred_improvement + 1e-20)

            assert not is_infnan(rho), "rho is not a valid number in adjust_radius code. Try disabling adjust_radius=False"
            if rho > self.eta0:      # accept new parameters
                break

            if self.adjusted_trust_radius < self.trust_radius_min:
                if self.verbose: self.msg(f"trust_radius_min={self.trust_radius_min} reached!", t)
                self.adjusted_trust_radius = self.trust_radius_min
                break

            if rho < self.eta1:
                self.adjusted_trust_radius *= 0.25
                if self.verbose > 1: self.msg(f"reducing adjusted_trust_radius to {self.adjusted_trust_radius} in trust radius iteration={tr_iter}", t)

            elif rho > self.eta2 and delta_x@delta_x >= self.adjusted_trust_radius**2:
                self.adjusted_trust_radius = min(2.0 * self.adjusted_trust_radius, self.trust_radius_max)
                if self.verbose > 1: self.msg(f"increasing adjusted_trust_radius to {self.adjusted_trust_radius} in trust radius iteration={tr_iter}", t)


        else: # for loop finished without breaking
            if self.verbose: self.msg(f"max_iter reached in adjusted_trust_radius loop!", t)

        self.f_last_trn = f_new_trn
        return new_x, f_new_trn


OPTIMIZERS = {'GradientDescent':GradientDescent, 
              'GradientDescentBB':GradientDescentBB, 
              'Adam':Adam,
              'NewtonMethod':NewtonMethod, 
              'NewtonMethodTrustRegion': NewtonMethodTrustRegion, 
              'TRON':TRON}


# This is the solution object that is returned by the optimize function
Solution = namedtuple('Solution', ['objective', 'x', 'val_objective'], defaults=[None])

def optimize(x0, objective, minimize=True, validation=None, 
             patience=10, max_iter=None,
             optimizer='GradientDescentBB', optimizer_kwargs=None,
             max_trn_objective=None, max_val_objective=None, min_trn_objective=None, min_val_objective=None,
             verbose=0, report_every=10, skip_max_iter_warning=False
             ):
    # Optimize function using the specified optimizer
    # Arguments:
    #   x0               : torch tensor or numpy array specifiying initial parameters
    #   objective        : Instance of Objective class that provides get_objective, get_gradient and/or get_hessian information
    #   minimize         : whether to minimize or maximize the objective function
    #   validation       : if not None, we use this Objective instance for early stopping. It should provide get_objective method
    #   patience (int)   : number of iterations to wait for validation improvement before stopping
    #   max_iter (int)   : maximum number of iterations (if None, uses default from optimizer)
    #   max_trn_objective (float) : maximum training objective value, clip and stop if exceeded
    #   max_val_objective (float) : maximum validation objective value, clip and stop if exceeded
    #   min_trn_objective (float) : minimum training objective value, clip and stop if below
    #   min_val_objective (float) : minimum validation objective value, clip and stop if below
    #   optimizer (str)  : Instance of Optimizer class to use. Should support .reset_state() and .get_update() methods
    #                      If None, we use the default optimizer (GradientDescentBB)
    #   verbose (int)    : Verbosity level for printing debugging information (0=no printing)
    #   report_every (int) : if verbose > 1, we report every report_every iterations
    #   skip_max_iter_warning (bool) : if True, we skip the warning about max_iter being reached
    #
    # Returns:
    #   Solution object with the following attributes:
    #       objective     : final objective value on training data set
    #       x             : vector of final parameters
    #       val_objective : final objective value on validation data set (if validation is not None)

    if type(optimizer) is str:
        optimizer = OPTIMIZERS[optimizer](verbose=verbose, minimize=minimize, **(optimizer_kwargs or {}))
    elif isinstance(optimizer, Optimizer):
        assert optimizer_kwargs is None, "optimizer_kwargs should be None if optimizer is an instance of Optimizer class"
        optimizer.minimize = minimize
        optimizer.reset_state()
    else:
        raise ValueError(f"Optimizer must be a string or an instance of Optimizer class, not {type(optimizer)}")
    
    run_max_iter = max_iter if max_iter is not None else optimizer.default_max_iter

    x = objective.initialize_parameters(x0)

    f_cur_trn = f_new_trn = np.nan
    if validation is not None:
        f_new_val = f_best_val = np.nan
        best_val_x       = x + 0 # make a copy
        best_val_iter    = 0
        patience_counter = 0
    
    old_time         = time.time()

    for t in range(run_max_iter):
        pfx = f"[t={t} trn={f_cur_trn:6.3f}" + (f" val={f_best_val:6.3f}" if validation is not None else "") + "]"
        r = optimizer.get_update(t=t, objective=objective, x=x)
        
        # get_update can return either new_x or (new_x, f_new_trn)
        if isinstance(r, Iterable) and len(r) == 2:  
            new_x, f_new_trn = r
        else:
            new_x = r
            if new_x is not None:
                f_new_trn  = objective.get_objective(x) 

        if new_x is None:
            if verbose: optimizer.msg(f"{pfx} [Stopping] Invalid update, x={x}")
            break

        if is_infnan(f_new_trn):
            if verbose: optimizer.msg(f"{pfx} [Stopping] Invalid training objective {f_new_trn}")
            break

        if verbose and verbose > 1 and report_every > 0 and t % report_every == 0:
            new_time = time.time()
            optimizer.msg(f'{(new_time - old_time)/report_every:4.2f}s/iter | f_new_trn={f_new_trn: 14.10f}' + 
                (f' f_new_val={f_new_val: 14.10f} f_best_val={f_best_val: 14.10f} patience_counter={patience_counter}' if validation is not None else '') )
            old_time = new_time

        if validation is not None:
            f_new_val = validation.get_objective(new_x) 
            if is_infnan(f_new_val):
                if verbose: optimizer.msg(f"{pfx} [Stopping] Invalid validation objective {f_new_val}")
                break

        x = new_x

        if max_trn_objective is not None and f_new_trn > max_trn_objective:
            f_cur_trn = max_trn_objective
            if verbose: optimizer.msg(f"{pfx} [Clipping] f_new_trn > max_trn_objective, ‖x‖∞={l1norm(x):3.2f}")
            break 

        if min_trn_objective is not None and f_new_trn < min_trn_objective:
            f_cur_trn = min_trn_objective
            if verbose: optimizer.msg(f"{pfx} [Clipping] f_new_trn < min_trn_objective, ‖x‖∞={l1norm(x):3.2f}")
            break 

        if abs(f_new_trn - f_cur_trn) < optimizer.tol:
            f_cur_trn = f_new_trn
            if verbose: optimizer.msg(f"{pfx}[Converged] Training objective change below tol={optimizer.tol}")
            break 

        f_cur_trn = f_new_trn

        if validation is not None: 
            if max_val_objective is not None and f_new_val > max_val_objective:
                f_best_val = max_val_objective
                best_val_x = x
                if verbose: optimizer.msg(f"{pfx} [Clipping] f_new_val > max_val_objective, ‖x‖∞={l1norm(x):3.2f}")
                break

            if min_val_objective is not None and f_new_val < min_val_objective:
                f_best_val = min_val_objective
                best_val_x = x
                if verbose: optimizer.msg(f"{pfx} [Clipping] f_new_val < min_val_objective, ‖x‖∞={l1norm(x):3.2f}")
                break

            improved = is_infnan(f_best_val) or (f_new_val < f_best_val if minimize else f_new_val > f_best_val)
            if improved:
                if verbose > 1 and t-best_val_iter>1: 
                    optimizer.msg(f"{pfx} [Patience] Resetting patience counter, last improvement {t-best_val_iter} steps ago")
                f_best_val       = f_new_val
                best_val_x       = x.clone()
                patience_counter = 0
                best_val_iter    = t

            elif patience_counter >= patience:
                if verbose:
                    optimizer.msg(f"{pfx} [Stopping] Validation objective did not improve for {patience} steps (last improvement {t-best_val_iter} steps ago)")
                break
            
            else:
                patience_counter += 1
        
                
    else:   # for loop did not break, so we reached Optimizers default_max_iter
        if max_iter is None and verbose:
            optimizer.msg(f'default_max_iter {max_iter} reached before convergence. ' + 
                            ('May want to increase max_iter' if not skip_max_iter_warning else ''))

    if validation is not None:
        return Solution(objective=f_cur_trn, x=best_val_x, val_objective=f_best_val)
    else:
        return Solution(objective=f_cur_trn, x=x, val_objective=None)



