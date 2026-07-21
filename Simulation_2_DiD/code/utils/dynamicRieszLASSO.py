import numpy as np
from scipy.stats import norm
import torch
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
import pdb 

# c1_vals = torch.tensor([5/4, 1, 3/4, 1/2]) / 2     # values from the ECTA paper
c1_vals = torch.tensor([1, 1/2, 1/4, 1/8, 1/16]) / 2     # I found that lower values work better

lasso_cv_settings_global = {
    'b_degree' : 1,
    'cv_folds' : 5,
    'random_state' : 42
}

lasso_a_settings_global = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' : "CV",
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

lasso_f_settings_global = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' : "CV",
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

def two_norm(x):
    x = x.squeeze()
    return torch.sqrt(torch.dot(x, x))

def m(X, d, a_prev, b_func):
    
    """
    M function for the dynamic riesz. We have that M_t(X,D,gamma) = a_t-1 (X_t-1, D_t-1) gamma (X_t, d_t)
    
    X (nx(p1...pt)) : History of covariates
    d (nxt) : History of counterfactual treatments

    a_prev (nx1) : The previous iteration of the RR function. a_(t-1) ( X_t-1, D_t-1 ). Not the function, but the actual predicted values.
    
    b_func (function (X, D)) : The LASSO mapping function
    """

    return a_prev * b_func(X, d)

def get_MG(X, D, d, a_prev, b_func):
    
    """
    Computes the M and G matrices for LASSO optimisation
    
    X (nx(p1...pt)) : History of covariates
    D (nxt) : History of treatments
    d (nxt) : History of counterfactual treatments
    a_prev (nx1) : The previous iteration of the RR function. a_(t-1) ( X_t-1, D_t-1 ). Not the function, but the actual predicted values.

    b_func (function (X, D)) : The LASSO mapping function
    
    """
    
    B = b_func(X, D)
    M_hat = torch.mean(m(X, d, a_prev, b_func),0).reshape(-1,1)
    G_hat = B.t() @ B / X.size(0)

    return M_hat, G_hat

def get_D(X, D, d, a_prev, b_func, rho_hat): #Here I have removed the m function from inputs, hoping that it recognizes the one above
    
    """
    Function for the LASSO program. 
    The D in the LASSO (different from treatment D) is 
    
    D (nxt) : History of treatment assignments
    d (nxt) : History of counterfactual treatments
    X (nx(p1...pt)) : History of covariates
    a_prev (nx1) : The previous iteration of the RR function. a_(t-1) ( X_t-1, D_t-1 ). Not the function, but the actual predicted values.
    
    b_func (function (X, D)) : The LASSO mapping function
    rho_hat (px1) : the current parameters of the LASSO 
    """

    b_val = b_func(X,D)

    term = ((b_val * (b_val * rho_hat.t()).sum(1).reshape(-1,1)) - m(X, d, a_prev, b_func)) ** 2
    
    return torch.sqrt(torch.mean(term,0).reshape(-1,1))  

def RMD_lasso(M, G, D_matrix, lambda_val=0, control={'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}, beta_start=None, c3=0.1):

    Gt = G.numpy()
    Mt = M.numpy()
    D_matrix = D_matrix.numpy()
    
    p = Gt.shape[1]
    L = torch.hstack((torch.tensor(c3).float(),torch.ones(p-1))).reshape(-1,1)
    lambda_vec = (lambda_val * L * D_matrix).numpy()
    
    beta = np.zeros((p,1)) if (beta_start == None) else beta_start
    
    wp = []
    mm = 1
    
    while mm < control['maxIter']:
        beta_old = beta.copy()
        
        for j in range(p):
            rho = Mt[j] - np.dot(Gt[j, :], beta) + Gt[j, j] * beta[j]
            z = Gt[j, j]
            
            if np.isnan(rho).any():
                beta[j] = 0
                continue
            
            if rho < -lambda_vec[j]:
                beta[j] = (rho + lambda_vec[j]) / z
            elif abs(rho) <= lambda_vec[j]:
                beta[j] = 0
            else:
                beta[j] = (rho - lambda_vec[j]) / z
        
        wp.append(beta.copy())
        
        if np.sum(np.abs(beta - beta_old)) < control['optTol']:
            break
        
        mm += 1
        
    w = beta
    w[np.abs(w) < control['zeroThreshold']] = 0

    return {'coefficients': torch.tensor(beta).float(), 'coef_list': wp, 'num_it': mm}

def RMD_stable(X, D, d, a_prev, b_func, D_LB, D_add, max_iter, c1, c2, tol, control, beta_start):
    
    n = D.shape[0]
    p = b_func(X[0], D[0]).shape[1]
    
    p0 = int(np.ceil(X.shape[1] / 4)) 

    X0 = X[:, :p0]
    M_hat0, G_hat0 = get_MG(X0, D, d, a_prev, b_func)    
        
    rho_hat = torch.linalg.lstsq(G_hat0, M_hat0.float()).solution
    rho_hat = torch.vstack((rho_hat, torch.zeros(p - G_hat0.shape[0],1)))
    
    M_hat, G_hat = get_MG(X, D, d, a_prev, b_func)
    
    lambda_val = c1 * norm.ppf(1 - c2 / (2 * p)) / np.sqrt(n)
    
    k = 1
    diff_rho = 1
    while diff_rho > tol and k <= max_iter:
        rho_hat_old = rho_hat.clone()
        D_hat_rho = get_D(X, D, d, a_prev, b_func, rho_hat_old)
        D_hat_rho = np.maximum(D_LB, D_hat_rho) + D_add

        rho_hat = RMD_lasso(M_hat, G_hat, D_hat_rho, lambda_val, control, beta_start)['coefficients']

        diff_rho = two_norm(rho_hat - rho_hat_old)
        k += 1
    
    return rho_hat

def get_optimal_c1(X, D, d, a_prev, b_func, D_LB, D_add, max_iter, c1_vals, c2, tol, control, beta_start, seed = None):
    """
    Function to find the optimal hyperparameter `c` using cross-validation.
    """
    n_vals = len(c1_vals)
    cv_vals = torch.empty(n_vals)
    
    for i, c1_val in enumerate(c1_vals):
        cv_vals[i] = crossval_c1(X, D, d, a_prev, b_func, D_LB, D_add, max_iter, c1_val, c2, tol, control, beta_start, seed = seed)
    
    # Find the value of `c` that minimizes the cross-validation loss
    c_star = c1_vals[torch.argmin(cv_vals)]

    return c_star

def crossval_c1(X, D, d, a_prev, b_func, D_LB, D_add, max_iter, c1, c2, tol, control, beta_start, seed = None):
    """
    Cross-validation function for ridge regression or conditional expectation function.

    For now: number of folds fixed to 5
    """
    cv_loss = []
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    # Iterate through folds
    for fold, (train_index, test_index) in enumerate(kf.split(X)):        
        
        d_l, d_nl = d[test_index], d[train_index]
        D_l, D_nl = D[test_index], D[train_index]
        X_l, X_nl = X[test_index, :], X[train_index, :]
        a_prev_l, a_prev_nl = a_prev[test_index, :], a_prev[train_index, :]
        
        # Get stage 1 (on training set)
        rho_nl = RMD_stable(X_nl, D_nl, d_nl, a_prev_nl, b_func, D_LB, D_add, max_iter, c1, c2, tol, control, beta_start)
        
        # Get cross-validation loss (on validation set)
        M_l, G_l = get_MG(X_l, D_l, d_l, a_prev_l, b_func)
        
        cv_loss_l = -2 * M_l.T @ rho_nl + rho_nl.T @ G_l @ rho_nl
        
        cv_loss.append(cv_loss_l.item())
    
    cv = torch.tensor(cv_loss).mean()
    return cv

class b_polynomial: 
    """
    The b function
    Class because of standardization
    """
    
    def __init__(self, degree):
        
        self.degree = degree
        self.standardization = None
        
    def b_poly(self, X, D):

        """
        Generates polynomial and interaction terms for LASSO regression.

        Parameters:
        Input: torch matrices
        """

        if X.ndim <= 1: #Only one observation, dimension of input is a one-dimensional vector (,p) -> transform to one row (1,p)
            X = X.reshape(1,-1)
            D = D.reshape(1,-1)

        # Generate interactions for X
        poly = PolynomialFeatures(degree=self.degree, interaction_only=False, include_bias=False)
        X_poly = torch.tensor(poly.fit_transform(X)).float()

        #Generate interactions between D:
        n_D = D.shape[1]  # Number of features from D
        D_interacted = D.clone()
        if n_D > 1:
            for r in range(2, n_D + 1):
                for comb in combinations(range(n_D), r):
                    # Create the interaction term by multiplying the selected columns
                    interaction_term = torch.ones(D.shape[0],1)
                    for idx in comb:
                        interaction_term *= D[:, idx:idx+1]
                    D_interacted = torch.hstack((D_interacted, interaction_term))
#
        # Now interact all X polynomials with all D interactions terms
        interactions_XD = (X_poly.unsqueeze(2) * D_interacted.unsqueeze(1)).reshape(X_poly.shape[0],-1)

        # Check for collinearity in interactions
        #columns_fully_zero = ~(interactions_XD.abs().sum(dim=0) == 0)
        #columns_equal_vector = ~torch.all(interactions_XD == D, dim=0)
        #interactions_XD = interactions_XD[:, columns_fully_zero & columns_equal_vector]



        full_covariates = torch.hstack((X_poly, D_interacted, interactions_XD))

        if (self.standardization is None):
            return full_covariates
        else:
            full_covariates = (full_covariates - self.standardization['mean'][:full_covariates.shape[1]]) / self.standardization['std'][:full_covariates.shape[1]]
            return torch.hstack((torch.ones(D.shape[0],1), full_covariates)) 
                
    def set_standardization(self, X, D):
        XD = self.b_poly(X, D)
        self.standardization = {'mean' : torch.zeros(XD.shape[1]), 'std' : torch.ones(XD.shape[1]) }
        return self
        
class Learner_a_LASSO:

    """
    This class estimates the a_t function:
    E[ a_t-1 (X_t-1, D_t-1) f_t(X_t, d_t) ] = E[ a_t (X_t, D_t) f_t(X_t, D_t) ]

    Note that for notation, any variable here contains it's whole history. E.g. X_t = (X_1, X_2, ..., X_t-1, X_t)

    Assume all input and output is alsways torch tensor
    """

    
    def __init__(self, lasso_a_settings = lasso_a_settings_global):
        """
        Parameters
        ----------
        """
                
        self.lambda_val = lasso_a_settings['lambda_val']
        self.beta_start = lasso_a_settings['beta_start']
        self.D_LB = lasso_a_settings['D_LB']
        self.D_add = lasso_a_settings['D_add']
        self.c1 = lasso_a_settings['c1']
        self.c2 = lasso_a_settings['c2']
        self.tol = lasso_a_settings['tol']
        self.max_iter = lasso_a_settings['max_iter']
        self.b_degree = lasso_a_settings['b_degree']
        self.control = lasso_a_settings['control']
        self.seed = lasso_a_settings['seed']

    def fit(self, X, D, a_prev):
        """
        Parameters
        ----------
        f_t (function (X,D)) : current prediction function f_t 
        
        X (nx(p1,...pT) : history of all covariates up to time t
        D (nxt) : history of all treatment assignments up to time t
        d (nxt) : history of all treatment counterfactuals up to time t
        
        a_prev (nx1) : a_t-1 (X_t-1, D_t-1) the previously step RR. Not the function, but the actual predicted values
        """

        d = D * 0
        
        # Standardization:
        b_func_class = b_polynomial(self.b_degree)
        b_func_class.set_standardization(X,D)
        self.b_func = b_func_class.b_poly


        # Tuning c1 
        if self.c1 == "CV":
            c1 = get_optimal_c1(X, D, d, a_prev, self.b_func, self.D_LB, self.D_add, self.max_iter, c1_vals, self.c2, self.tol, self.control, self.beta_start, seed = self.seed)
            self.c1 = c1
        
        # Apply LASSO
        self.rho_hat = RMD_stable(X, D, d, a_prev, self.b_func, self.D_LB, self.D_add, self.max_iter, self.c1, self.c2, self.tol, self.control, self.beta_start)
        
        return self

    def predict(self, X, D):
        """
        Parameters
        ----------
        X (nx(p1,...pT) : history of all covariates up to time t.
        D (nxt) : history of all treatment assignments up to time t.
        """

        return torch.sum(self.b_func(X,D) * self.rho_hat.t(),1).reshape(-1,1)
      
class Learner_f_LASSO:

    """
    This class estimates the f_t function using LASSO:
    f_t(X_t, D_t) = E [ f_t+1(X_t+1, d_t+1) | X_t, D_t ].

    Note that for notation, any variable here contains it's whole history. E.g. X_t = (X_1, X_2, ..., X_t-1, X_t)

    Assume all input and output is alsways torch tensor
    """
    
    def __init__(self, lasso_f_settings = lasso_f_settings_global):
        """
        Parameters
        ----------
        """
                
        self.lambda_val = lasso_f_settings['lambda_val']
        self.beta_start = lasso_f_settings['beta_start']
        self.D_LB = lasso_f_settings['D_LB']
        self.D_add = lasso_f_settings['D_add']
        self.c1 = lasso_f_settings['c1']
        self.c2 = lasso_f_settings['c2']
        self.tol = lasso_f_settings['tol']
        self.max_iter = lasso_f_settings['max_iter']
        self.b_degree = lasso_f_settings['b_degree']
        self.control = lasso_f_settings['control']
        self.seed = lasso_f_settings['seed']
    
    def fit(self, Y, X, D):
        """
        Parameters
        ----------
        f_t (function (X,D)) : current prediction function f_t 
        
        X (nx(p1,...pT) : history of all covariates up to time t
        D (nxt) : history of all treatment assignments up to time t
        d (nxt) : history of all treatment counterfactuals up to time t
        
        a_prev (nx1) : a_t-1 (X_t-1, D_t-1) the previously step RR. Not the function, but the actual predicted values
        """
        
        # Standardization:
        b_func_class = b_polynomial(self.b_degree)
        b_func_class.set_standardization(X,D)
        self.b_func = b_func_class.b_poly

        # Tuning c1 
        if self.c1 == "CV":
            c1 = get_optimal_c1(X, D, D, Y, self.b_func, self.D_LB, self.D_add, self.max_iter, c1_vals, self.c2, self.tol, self.control, self.beta_start, seed = self.seed)
            self.c1 = c1
        
        # Apply LASSO
        self.rho_hat = RMD_stable(X, D, D, Y, self.b_func, self.D_LB, self.D_add, self.max_iter, self.c1, self.c2, self.tol, self.control, self.beta_start)
        
        return self

    
    def predict(self, X, D):
        """
        Parameters
        ----------
        X (nx(p1,...pT) : history of all covariates up to time t.
        D (nxt) : history of all treatment assignments up to time t.
        """

        return torch.sum(self.b_func(X,D) * self.rho_hat.t(),1).reshape(-1,1)
    