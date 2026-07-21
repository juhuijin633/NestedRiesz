import numpy as np
import torch
from econml.grf._base_grf import BaseGRF
from econml.utilities import cross_product
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

rf_a_settings_global = {
    'poly_degree' : 0,
    'l2' : 0,
    'n_estimators' : 100,
    'criterion' : "mse",
    'max_depth' : None,
    'min_samples_split' : 10,
    'min_samples_leaf' : 5,
    'min_weight_fraction_leaf' : 0.,
    'min_var_fraction_leaf' : None,
    'min_var_leaf_on_val' : False,
    'max_features' : "auto",
    'min_impurity_decrease' : 0.,
    'max_samples' : .45,
    'min_balancedness_tol' : .45,
    'honest' : True,
    'inference' : True,
    'fit_intercept' : True,
    'subforest_size' : 4,
    'n_jobs' : -1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}
rf_f_settings_global = {
    'poly_degree' : 1,
    'l2' : 0,
    'n_estimators' : 100,
    'criterion' : "mse",
    'max_depth' : None,
    'min_samples_split' : 10,
    'min_samples_leaf' : 5,
    'min_weight_fraction_leaf' : 0.,
    'min_var_fraction_leaf' : None,
    'min_var_leaf_on_val' : False,
    'max_features' : "auto",
    'min_impurity_decrease' : 0.,
    'max_samples' : .45,
    'min_balancedness_tol' : .45,
    'honest' : True,
    'inference' : True,
    'fit_intercept' : True,
    'subforest_size' : 4,
    'n_jobs' : -1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}

def b_poly(X, D, degree = 1):

    """
    Generates polynomial and interaction terms for LASSO regression.

    Parameters:
    Input: torch matrices

    Important, for RF no interactions between X and D (don't know why but it breaks)
    Q: D vs D interacted for T=2?
    """

    if not torch.is_tensor(D):
        D = torch.tensor(D).float()  # Convert T to a tensor
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()


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

    if degree == 0:
        return torch.hstack((torch.ones(D.shape[0],1), D_interacted)).numpy()
    else:
            # Generate interactions for X
            poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
            X_poly = torch.tensor(poly.fit_transform(X)).float()

            interactions_XD = (X_poly.unsqueeze(2) * D_interacted.unsqueeze(1)).reshape(X_poly.shape[0],-1)

            return torch.hstack((torch.ones(D.shape[0],1), X_poly, D_interacted, interactions_XD)).numpy()

def b_poly_f(X, D, degree = 1):

    """
    Generates polynomial and interaction terms for LASSO regression.

    Parameters:
    Input: torch matrices

    """

    if not torch.is_tensor(D):
        D = torch.tensor(D).float()  # Convert T to a tensor
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()

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

    # Generate interactions for X
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    X_poly = torch.tensor(poly.fit_transform(X)).float()

    interactions_XD = (X_poly.unsqueeze(2) * D_interacted.unsqueeze(1)).reshape(X_poly.shape[0],-1)

    return torch.hstack((torch.ones(D.shape[0],1), X_poly, D_interacted, interactions_XD)).numpy()

def m(X, d, a_prev, b_func, poly_degree):
    
    """
    M function for the dynamic riesz. We have that M_t(X,D,gamma) = a_t-1 (X_t-1, D_t-1) gamma (X_t, d_t)
    
    X (nx(p1...pt)) : History of covariates
    d (nxt) : History of counterfactual treatments

    a_prev (nx1) : The previous iteration of the RR function. a_(t-1) ( X_t-1, D_t-1 ). Not the function, but the actual predicted values.
    
    b_func (function (X, D)) : The LASSO mapping function
    """

    return a_prev * b_func(X, d, poly_degree)

class Learner_a_RFrr(BaseGRF):
    
    def __init__(self, *, rf_a_settings = rf_a_settings_global):
        """
        """
        # Unpack the settings
        poly_degree = rf_a_settings['poly_degree']
        l2 = rf_a_settings['l2']
        n_estimators = rf_a_settings['n_estimators']
        criterion = rf_a_settings['criterion']
        max_depth = rf_a_settings['max_depth']
        min_samples_split = rf_a_settings['min_samples_split']
        min_samples_leaf = rf_a_settings['min_samples_leaf']
        min_weight_fraction_leaf = rf_a_settings['min_weight_fraction_leaf']
        min_var_fraction_leaf = rf_a_settings['min_var_fraction_leaf']
        min_var_leaf_on_val = rf_a_settings['min_var_leaf_on_val']
        max_features = rf_a_settings['max_features']
        min_impurity_decrease = rf_a_settings['min_impurity_decrease']
        max_samples = rf_a_settings['max_samples']
        min_balancedness_tol = rf_a_settings['min_balancedness_tol']
        honest = rf_a_settings['honest']
        inference = rf_a_settings['inference']
        fit_intercept = rf_a_settings['fit_intercept']
        subforest_size = rf_a_settings['subforest_size']
        n_jobs = rf_a_settings['n_jobs']
        random_state = rf_a_settings['random_state']
        verbose = rf_a_settings['verbose']
        warm_start = rf_a_settings['warm_start']
        
        
        #Initialize the RF settings
        self.riesz_feature_fns = b_poly
        self.poly_degree = poly_degree
        self.moment_fn = m
        self.l2 = l2

        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=min_var_fraction_leaf,
                         min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features,
                         min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples,
                         min_balancedness_tol=min_balancedness_tol,
                         honest=honest,
                         inference=inference,
                         fit_intercept=False,
                         subforest_size=subforest_size,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start)

    def _get_alpha_and_pointJ(self, X, T, y, **kwargs):

        d = kwargs.get('d', None)  # Default to None if not provided
        a_prev = kwargs.get('a_prev', None)    # Default to 0 if not provided

        riesz_feats = self.riesz_feature_fns(X, T, self.poly_degree)
        mfeats = self.moment_fn(X, d, a_prev, self.riesz_feature_fns, self.poly_degree)        
        # riesz_cov_matrix = cross_product(riesz_feats, riesz_feats).reshape(X.shape[0],-1)

        n_riesz_feats = riesz_feats.shape[1]
        riesz_cov_matrix = cross_product(riesz_feats, riesz_feats).reshape(
            (X.shape[0], n_riesz_feats, n_riesz_feats)) + self.l2 * np.eye(n_riesz_feats)

        return mfeats, riesz_cov_matrix.reshape((X.shape[0], -1))

    def _get_n_outputs_decomposition(self, X, T, y, **kwargs):
        n_relevant_outputs = self.riesz_feature_fns(X, T, self.poly_degree).shape[1]
        n_outputs = n_relevant_outputs
        return n_outputs, n_relevant_outputs

    def _translate(self, point, X, T, **kwargs):

        riesz_feats = self.riesz_feature_fns(X, T, self.poly_degree)
        n_riesz_feats = riesz_feats.shape[1]

        riesz = np.sum(point[:, :n_riesz_feats] * riesz_feats, axis=1)
        return riesz

    def predict_riesz(self, X, T, interval=False, alpha=0.05, **kwargs):
        # TODO. the confidence interval for reg is not exactly accurate as
        # for T=1 it is the sum of two parameters and so we need to use
        # the variance of this sum and not the sum of the lower and upper ends

        # TODO. Maybe T_test should also be passed explicitly and not as the first coordinate
        # of X_test. Now there is inconsistency between the fit and predict API
        if interval:
            point, lb, ub = self.predict(
                X[:, 1:], interval=interval, alpha=alpha)
            riesz  = self._translate(point, X)
            lb_riesz = self._translate(lb, X)
            ub_riesz = self._translate(ub, X)
            return (riesz, lb_riesz, ub_riesz)
        else:
            point = self.predict(X, interval=interval, alpha=alpha)
            return torch.tensor(self._translate(point, X, T)).float().reshape(-1,1)

class Learner_a_RF:

    
    def __init__(self, rf_a_settings = rf_a_settings_global):
        """
        Parameters
        ----------
        b_function (function(X,D)) : the LASSO dictionary function that we use. Maps np matrices X and D into a single numpy matrix (with transformations)
        control : dictionary of settings for the optimization ( control={'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}, beta_start=None )
        lambda_val=0,
        beta_start=None
        """
                
        self.rf_a_settings = rf_a_settings

    def fit(self, X, D, d, a_prev):
        """
        Parameters
        ----------
        f_t (function (X,D)) : current prediction function f_t 
        
        X (nx(p1,...pT) : history of all covariates up to time t
        D (nxt) : history of all treatment assignments up to time t
        d (nxt) : history of all treatment counterfactuals up to time t
        
        a_prev (nx1) : a_t-1 (X_t-1, D_t-1) the previously step RR. Not the function, but the actual predicted values
        """

        X = X.numpy()
        D = D.numpy()
        d = d.numpy()
        a_prev = a_prev.numpy()
        y = torch.ones(X.shape[0], 1)

        # # # Tuning l2 
        # if self.l2 == "CV":
        #     l2 = get_optimal_c1(X, D, d, a_prev, c1_vals)
        #     self.l2 = l2
        
        # Apply RF
        self.learner = Learner_a_RFrr(rf_a_settings = self.rf_a_settings)
        self.learner.fit(X, D, y, d = d, a_prev = a_prev)

        return self

    def predict(self, X, D):
        """
        Parameters
        ----------
        X (nx(p1,...pT) : history of all covariates up to time t.
        D (nxt) : history of all treatment assignments up to time t.
        """

        return self.learner.predict_riesz(X, D)
  
class Learner_f_RF:
    
    def __init__(self, rf_f_settings = rf_f_settings_global):
        """
        Parameters
        ----------
        """
                
        self.rf_f_settings = rf_f_settings

    def fit(self, y, X, D):
        """
        """

        X = X.numpy()
        D = D.numpy()
        y = y.numpy()
        
        # Apply RF
        self.learner = Learner_f_RFreg(rf_f_settings=self.rf_f_settings)
        self.learner.fit(X, D, y)

        return self

    def predict(self, X, D):
        """
        Parameters
        ----------
        X (nx(p1,...pT) : history of all covariates up to time t.
        D (nxt) : history of all treatment assignments up to time t.
        """

        return self.learner.predict_reg(X, D)
  
class Learner_f_RFreg(BaseGRF):

    def __init__(self, *, rf_f_settings = rf_f_settings_global):
        """
        """
        # Unpack the settings
        poly_degree = rf_f_settings['poly_degree']
        l2 = rf_f_settings['l2']
        n_estimators = rf_f_settings['n_estimators']
        criterion = rf_f_settings['criterion']
        max_depth = rf_f_settings['max_depth']
        min_samples_split = rf_f_settings['min_samples_split']
        min_samples_leaf = rf_f_settings['min_samples_leaf']
        min_weight_fraction_leaf = rf_f_settings['min_weight_fraction_leaf']
        min_var_fraction_leaf = rf_f_settings['min_var_fraction_leaf']
        min_var_leaf_on_val = rf_f_settings['min_var_leaf_on_val']
        max_features = rf_f_settings['max_features']
        min_impurity_decrease = rf_f_settings['min_impurity_decrease']
        max_samples = rf_f_settings['max_samples']
        min_balancedness_tol = rf_f_settings['min_balancedness_tol']
        honest = rf_f_settings['honest']
        inference = rf_f_settings['inference']
        fit_intercept = rf_f_settings['fit_intercept']
        subforest_size = rf_f_settings['subforest_size']
        n_jobs = rf_f_settings['n_jobs']
        random_state = rf_f_settings['random_state']
        verbose = rf_f_settings['verbose']
        warm_start = rf_f_settings['warm_start']
        
        #Initialize the RF settings
        self.reg_feature_fns = b_poly
        self.poly_degree = poly_degree
        self.l2 = l2

        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=min_var_fraction_leaf,
                         min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features,
                         min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples,
                         min_balancedness_tol=min_balancedness_tol,
                         honest=honest,
                         inference=inference,
                         fit_intercept=False,
                         subforest_size=subforest_size,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start)

    def _get_alpha_and_pointJ(self, X, T, y):

        reg_feats = self.reg_feature_fns(X, T, self.poly_degree)
        n_reg_feats = reg_feats.shape[1]
        alpha = y.reshape(-1, 1) * reg_feats
        
        reg_cov_matrix = cross_product(reg_feats, reg_feats).reshape(
            (X.shape[0], n_reg_feats, n_reg_feats)) + self.l2 * np.eye(n_reg_feats)
        
        return alpha, reg_cov_matrix.reshape((X.shape[0], -1))

    def _get_n_outputs_decomposition(self, X, T, y):
        n_relevant_outputs = self.reg_feature_fns(X, T, self.poly_degree).shape[1]
        n_outputs = n_relevant_outputs
        return n_outputs, n_relevant_outputs

    def _translate(self, point, X, T):

        reg_feats = self.reg_feature_fns(X, T, self.poly_degree)
        
        reg = np.sum(point * reg_feats, axis=1)

        return reg

    def predict_reg(self, X, T, interval=False, alpha=0.05):
        # TODO. the confidence interval for reg is not exactly accurate as
        # for T=1 it is the sum of two parameters and so we need to use
        # the variance of this sum and not the sum of the lower and upper ends

        # TODO. Maybe T_test should also be passed explicitly and not as the first coordinate
        # of X_test. Now there is inconsistency between the fit and predict API
        if interval:
            point, lb, ub = self.predict(
                X[:, 1:], interval=interval, alpha=alpha)
            reg = self._translate(point, X)
            lb_reg = self._translate(lb, X)
            ub_reg = self._translate(ub, X)
            return (reg, lb_reg, ub_reg)
        else:
            point = self.predict(X, interval=interval, alpha=alpha)
            return torch.tensor(self._translate(point, X, T)).float().reshape(-1,1)

class Learner_f_RF_RFReg:

    """
    This class estimates the f_t function using sklearn RandomForestRegressor.
    f_t(X_t, D_t) = E [ f_t+1(X_t+1, d_t+1) | X_t, D_t ].

    Note that for notation, any variable here contains it's whole history. E.g. X_t = (X_1, X_2, ..., X_t-1, X_t)

    Assume all input and output is alsways torch tensor
    """


    def __init__(self, n_estimators = 100, random_state = 42, rf_f_settings = rf_f_settings_global):
        
        self.n_estimators = n_estimators
        self.random_state = random_state


    def fit(self, y, X, D):
        """
        Parameters
        ----------
        y (nx1) : outcome variable, which is f_t+1(X_t+1, d_t+1). 
        X (nx(p1,...pT) : history of all covariates up to time t.
        D (nxt) : history of all treatment assignments up to time t.
        """
        
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()  
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()  
        if isinstance(D, torch.Tensor):
            D = D.detach().cpu().numpy()  
            
        XD = np.hstack((X, D))
            
        self.f = RandomForestRegressor(n_estimators = self.n_estimators, random_state = self.random_state)
        
        self.f.fit(XD, y.ravel())
        

        return self
    
    def predict(self, X, D):
        """
        Parameters
        ----------
        X (nx(p1,...pT) : history of all covariates up to time t.
        D (nxt) : history of all treatment assignments up to time t.
        """
        
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()  
        if isinstance(D, torch.Tensor):
            D = D.detach().cpu().numpy()  

        XD = np.hstack((X, D))

        return torch.tensor(self.f.predict(XD).reshape(-1,1)).float()





# c1_vals = torch.tensor([0.1, 0.01, 0.005, 0.001])    
# def get_optimal_c1(X, D, d, a_prev, c1_vals):
#     """
#     Function to find the optimal hyperparameter `c` using cross-validation.
#     """
#     n_vals = len(c1_vals)
#     cv_vals = torch.empty(n_vals)
    
#     for i, c1_val in enumerate(c1_vals):
#         cv_vals[i] = crossval_c1(X, D, d, a_prev, c1_val)
    
#     # Find the value of `c` that minimizes the cross-validation loss
#     c_star = c1_vals[torch.argmin(cv_vals)]
#     c_star = c1_vals[torch.argmin(torch.abs(cv_vals))]

#     return c_star

# def crossval_c1(X, D, d, a_prev, l2):
#     """
#     Cross-validation function for ridge regression or conditional expectation function.

#     For now: number of folds fixed to 5
#     """
#     cv_loss = torch.zeros(X.shape[0],1)

#     folds = 5
#     kf = KFold(n_splits=folds, shuffle=True)
#     # Iterate through folds
#     for fold, (train_index, test_index) in enumerate(kf.split(X)):        
        
#         d_l, d_nl = d[test_index], d[train_index]
#         D_l, D_nl = D[test_index], D[train_index]
#         X_l, X_nl = X[test_index, :], X[train_index, :]
#         a_prev_l, a_prev_nl = a_prev[test_index, :], a_prev[train_index, :]
        
#         # Get stage 1 (on training set)
#         # rho_nl = RMD_stable(X_nl, D_nl, d_nl, a_prev_nl, b_func, D_LB, D_add, max_iter, c1, c2, tol, control, beta_start)

#         learner = Learner_a_RFrr(l2 = l2)
#         y_nl = torch.ones(X_nl.shape[0], 1)
#         learner.fit(X_nl, D_nl, y_nl, d = d_nl, a_prev = a_prev_nl)

#         a_hat = learner.predict_riesz(X_l, D_l)
#         a_hat_d = learner.predict_riesz(X_l, d_l)

#         cv_loss[test_index] = a_hat ** 2 - 2 * torch.tensor(a_prev_l) * a_hat_d

#         cv_loss[test_index] = torch.mean( a_hat * b_poly(X_l,D_l) - torch.tensor(a_prev_l) * b_poly(X_l,d_l) , 1 ).reshape(-1,1)
                
#     cv = torch.mean(cv_loss)
#     return cv

# class Learner_f_RF_old:

#     """
#     This class estimates the f_t function:
#     f_t(X_t, D_t) = E [ f_t+1(X_t+1, d_t+1) | X_t, D_t ].

#     Note that for notation, any variable here contains it's whole history. E.g. X_t = (X_1, X_2, ..., X_t-1, X_t)

#     Assume all input and output is alsways torch tensor
#     """

    
#     def __init__(self, n_estimators = 100, random_state = 42):
        
#         self.n_estimators = n_estimators
#         self.random_state = random_state


#     def fit(self, y, X, D):
#         """
#         Parameters
#         ----------
#         y (nx1) : outcome variable, which is f_t+1(X_t+1, d_t+1). 
#         X (nx(p1,...pT) : history of all covariates up to time t.
#         D (nxt) : history of all treatment assignments up to time t.
#         """
        
#         if isinstance(X, torch.Tensor):
#             X = X.detach().cpu().numpy()  
#         if isinstance(y, torch.Tensor):
#             y = y.detach().cpu().numpy()  
#         if isinstance(D, torch.Tensor):
#             D = D.detach().cpu().numpy()  
            
#         XD = np.hstack((X, D))
#         # XD = b_poly_f(X, D)
            
#         self.f = RandomForestRegressor(n_estimators = self.n_estimators, random_state = self.random_state)
        
#         self.f.fit(XD, y.ravel())
        

#         return self
    
#     def predict(self, X, D):
#         """
#         Parameters
#         ----------
#         X (nx(p1,...pT) : history of all covariates up to time t.
#         D (nxt) : history of all treatment assignments up to time t.
#         """
        
#         if isinstance(X, torch.Tensor):
#             X = X.detach().cpu().numpy()  
#         if isinstance(D, torch.Tensor):
#             D = D.detach().cpu().numpy()  

#         XD = np.hstack((X, D))
#         # XD = b_poly_f(X, D)

#         return torch.tensor(self.f.predict(XD).reshape(-1,1)).float()

