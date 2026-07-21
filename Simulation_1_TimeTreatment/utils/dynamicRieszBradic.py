import os

import numpy as np
from scipy.stats import norm
import torch
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LassoCV

from rpy2.robjects import r, default_converter
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects.packages import importr

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

import time
import pdb
# pdb.set_trace()  # Code will stop here

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

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
            for deg in range(2, n_D + 1):
                for comb in combinations(range(n_D), deg):
                    # Create the interaction term by multiplying the selected columns
                    interaction_term = torch.ones(D.shape[0],1)
                    for idx in comb:
                        interaction_term *= D[:, idx:idx+1]
                    D_interacted = torch.hstack((D_interacted, interaction_term))

        # Now interact all X polynomials with all D interactions terms
        interactions_XD = (X_poly.unsqueeze(2) * D_interacted.unsqueeze(1)).reshape(X_poly.shape[0],-1)

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
        
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr



def estimateBradic(Y, X, D, X_index, folds):

    # List of required R packages
    required_packages = ['glmnet', 'itertools', 'doParallel', 'RCAL']

    # Get R utils package
    utils = importr('utils')
    cran_mirror = "https://cloud.r-project.org"

    # Install missing packages
    for pkg in required_packages:
        is_installed = ro.r(f"is.element('{pkg}', installed.packages()[,1])")[0]
        if not is_installed:
            utils.install_packages(pkg, repos=cran_mirror)

    # Prepare R object for conversion
    r = ro.r

    # Prepare numpy arrays
    S1 = X[:, :X_index[0]+1].numpy()
    S2 = X[:, X_index[0]+1:X_index[1]+1].numpy()
    A1 = D[:, :1].numpy()
    A2 = D[:, 1:].numpy()
    Y = Y.numpy()

    # Use conversion context instead of numpy2ri.activate()
    with localconverter(ro.default_converter + numpy2ri.converter):
        r.assign("S1", S1)
        r.assign("S2", S2)
        r.assign("Y", Y)
        r.assign("A1", A1)
        r.assign("A2", A2)
        r.assign("K", folds)

    # Read the R script file
    bradic_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Bradic.R")
    with open(bradic_r, "r") as file:
        r_code = file.read()

    # Execute the R code
    result = r(r_code)

    return result

        
def estimateBradicT1(Y, X, D, folds):

    Y = Y.numpy()
    X = X.numpy()
    D = D.numpy()
    X_std = np.std(X, 0)
    X_std[X_std < 1e-8] = 1.0
    X = (X - np.mean(X, 0)) / X_std

    X = np.hstack((np.ones((X.shape[0],1)),X))  
      

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    n = Y.shape[0]
    fold_results = torch.zeros(n,1)

    pi1 = np.zeros(n)
    mu1_1 = np.zeros(n)
    mu1_0 = np.zeros(n)
    gamma1_1 = np.zeros(n)
    gamma1_0 = np.zeros(n)
    # Iterate through folds
    for fold, (train_index, test_index) in enumerate(kf.split(Y)):

        # Splitting data
        X_train, X_test = X[train_index], X[test_index]
        D_train, D_test = D[train_index], D[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Fit logistic regression with cross-validation
        fit_Log1 = LogisticRegressionCV(cv=5, solver='liblinear', penalty='l1', scoring='neg_log_loss', max_iter=1000)
        fit_Log1.fit(X_train, D_train.ravel())

        # Predict probabilities for the new data
        pi1[test_index] = fit_Log1.predict_proba(X_test)[:, 1]  # [:, 1] gives the probability of class 1

        rho_hat = fit_Log1.coef_

        # Fit LASSO regression for m_1
        fit_lasso1 = LassoCV(cv=5, random_state=42).fit(
            X_train[D_train.ravel() == 1], Y_train[D_train.ravel() == 1].ravel()
        )
        mu1_1[test_index] = fit_lasso1.predict((X_test))

        # Fit LASSO regression for m_0
        fit_lasso0 = LassoCV(cv=5, random_state=42).fit(
            X_train[D_train.ravel() == 0], Y_train[D_train.ravel() == 0].ravel()
        )
        mu1_0[test_index] = fit_lasso0.predict((X_test))


    pi1 = np.clip(pi1, 1e-6, 1 - 1e-6)
    gamma1_1 = D.ravel() / pi1
    pred1 = np.mean(gamma1_1 * Y.ravel() - (gamma1_1 - 1) * mu1_1  )
    sig1 = np.sqrt(np.mean((gamma1_1 * Y.ravel() - (gamma1_1 - 1) * mu1_1 - pred1) ** 2))
    
    gamma1_0 = (1 - D.ravel()) / (1 - pi1)
    pred0 = np.mean(gamma1_0 * Y.ravel() - (gamma1_0 - 1) * mu1_0  )
    sig0 = np.sqrt(np.mean((gamma1_0 * Y.ravel() - (gamma1_0 - 1) * mu1_0 - pred0) ** 2))
    # pdb.set_trace()  # Code will stop here

    return pred1 - pred0, pred1, pred0, sig1, sig0, torch.tensor(gamma1_1).reshape(-1,1), torch.tensor(gamma1_0).reshape(-1,1), torch.tensor(mu1_1).reshape(-1,1), torch.tensor(mu1_1).reshape(-1,1), rho_hat


